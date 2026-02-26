//! ZRTL Plugin Support
//!
//! Types and macros for creating ZRTL plugins that can be loaded by the Zyntax runtime.

use crate::TypeTag;
use std::ffi::c_char;

/// Current ZRTL format version
pub const ZRTL_VERSION: u32 = 1;

/// Maximum number of parameters in a function signature
pub const MAX_PARAMS: usize = 16;

/// Function signature for ZRTL symbols (C ABI compatible)
///
/// Describes parameter and return types for a plugin function.
/// The compiler uses this to determine whether to auto-box arguments.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ZrtlSymbolSig {
    /// Number of parameters
    pub param_count: u8,
    /// Flags for the function signature
    pub flags: ZrtlSigFlags,
    /// Return type (VOID if no return)
    pub return_type: TypeTag,
    /// Parameter types (up to MAX_PARAMS)
    pub params: [TypeTag; MAX_PARAMS],
}

impl ZrtlSymbolSig {
    /// Create an empty signature (no params, void return)
    pub const fn empty() -> Self {
        Self {
            param_count: 0,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::VOID,
            params: [TypeTag::VOID; MAX_PARAMS],
        }
    }

    /// Create a signature with the given return type and parameters
    pub const fn new(return_type: TypeTag, params: &[TypeTag]) -> Self {
        let mut sig = Self::empty();
        sig.return_type = return_type;
        sig.param_count = params.len() as u8;
        let mut i = 0;
        while i < params.len() && i < MAX_PARAMS {
            sig.params[i] = params[i];
            i += 1;
        }
        sig
    }

    /// Check if parameter at index expects DynamicBox
    pub fn param_is_dynamic(&self, index: usize) -> bool {
        if index >= self.param_count as usize {
            return false;
        }
        self.params[index].is_category(crate::TypeCategory::Opaque)
            || self.flags.contains(ZrtlSigFlags::ALL_DYNAMIC)
    }

    /// Check if return type is DynamicBox
    pub fn returns_dynamic(&self) -> bool {
        self.return_type.is_category(crate::TypeCategory::Opaque)
    }
}

/// Flags for function signatures
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZrtlSigFlags(pub u8);

impl ZrtlSigFlags {
    /// No flags
    pub const NONE: Self = Self(0);
    /// All parameters should be passed as DynamicBox
    pub const ALL_DYNAMIC: Self = Self(1 << 0);
    /// Function is variadic
    pub const VARIADIC: Self = Self(1 << 1);
    /// Function may have side effects
    pub const EFFECTFUL: Self = Self(1 << 2);

    /// Check if flag is set
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Combine flags
    pub const fn with(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }
}

/// Special TypeTag constant for DynamicBox parameter type
impl TypeTag {
    /// TypeTag indicating a DynamicBox parameter
    pub const DYNAMIC_BOX: Self =
        Self::new(crate::TypeCategory::Opaque, 0xFFFF, crate::TypeFlags::NONE);
}

/// Symbol entry in the ZRTL symbol table (C ABI compatible)
///
/// Matches the layout of `ZrtlSymbol` in `zrtl.h`.
#[repr(C)]
pub struct ZrtlSymbol {
    /// Symbol name (null-terminated C string)
    /// Convention: `$TypeName$method_name`
    pub name: *const c_char,
    /// Function pointer
    pub ptr: *const u8,
    /// Optional pointer to function signature (null if not provided)
    pub sig: *const ZrtlSymbolSig,
}

impl ZrtlSymbol {
    /// Create a null/sentinel symbol
    pub const fn null() -> Self {
        Self {
            name: std::ptr::null(),
            ptr: std::ptr::null(),
            sig: std::ptr::null(),
        }
    }

    /// Create a new symbol without signature
    pub const fn new(name: *const c_char, ptr: *const u8) -> Self {
        Self {
            name,
            ptr,
            sig: std::ptr::null(),
        }
    }

    /// Create a new symbol with signature
    pub const fn with_sig(name: *const c_char, ptr: *const u8, sig: *const ZrtlSymbolSig) -> Self {
        Self { name, ptr, sig }
    }

    /// Check if this symbol has signature information
    pub fn has_sig(&self) -> bool {
        !self.sig.is_null()
    }

    /// Get the signature if available
    pub unsafe fn get_sig(&self) -> Option<&ZrtlSymbolSig> {
        if self.sig.is_null() {
            None
        } else {
            Some(&*self.sig)
        }
    }
}

// SAFETY: ZrtlSymbol contains only immutable pointers to static data
unsafe impl Sync for ZrtlSymbol {}
unsafe impl Send for ZrtlSymbol {}

/// Plugin metadata (C ABI compatible)
///
/// Matches the layout of `ZrtlInfo` in `zrtl.h`.
#[repr(C)]
pub struct ZrtlInfo {
    /// ZRTL format version (must match ZRTL_VERSION)
    pub version: u32,
    /// Plugin name (null-terminated C string)
    pub name: *const c_char,
}

impl ZrtlInfo {
    /// Create plugin info
    pub const fn new(name: *const c_char) -> Self {
        Self {
            version: ZRTL_VERSION,
            name,
        }
    }
}

// SAFETY: ZrtlInfo contains only immutable data
unsafe impl Sync for ZrtlInfo {}
unsafe impl Send for ZrtlInfo {}

/// Symbol entry for the inventory crate
///
/// Used by the `#[zrtl_export]` attribute macro.
pub struct ZrtlSymbolEntry {
    /// Symbol name (with null terminator)
    pub name: &'static str,
    /// Function pointer
    pub ptr: *const u8,
}

// SAFETY: ZrtlSymbolEntry contains only static data
unsafe impl Sync for ZrtlSymbolEntry {}
unsafe impl Send for ZrtlSymbolEntry {}

// Register the inventory collector
inventory::collect!(ZrtlSymbolEntry);

/// Macro to define a ZRTL symbol table entry (legacy, no signature)
///
/// # Example
///
/// ```ignore
/// zrtl_symbol!("$Array$push", array_push)
/// ```
#[macro_export]
macro_rules! zrtl_symbol {
    ($name:expr, $func:ident) => {
        $crate::ZrtlSymbol::new(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
        )
    };
}

/// Macro to define a ZRTL symbol with signature
///
/// # Example
///
/// ```ignore
/// // Function that takes DynamicBox and returns DynamicBox
/// zrtl_symbol_sig!("$IO$print", io_print, dynamic(1) -> void)
/// zrtl_symbol_sig!("$Math$add", math_add, (i32, i32) -> i32)
/// ```
#[macro_export]
macro_rules! zrtl_symbol_sig {
    // All-dynamic signature (all params are DynamicBox)
    ($name:expr, $func:ident, dynamic($count:expr) -> void) => {{
        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: $count,
            flags: $crate::ZrtlSigFlags::ALL_DYNAMIC,
            return_type: $crate::TypeTag::VOID,
            params: [$crate::TypeTag::DYNAMIC_BOX; $crate::MAX_PARAMS],
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // All-dynamic signature with dynamic return
    ($name:expr, $func:ident, dynamic($count:expr) -> dynamic) => {{
        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: $count,
            flags: $crate::ZrtlSigFlags::ALL_DYNAMIC,
            return_type: $crate::TypeTag::DYNAMIC_BOX,
            params: [$crate::TypeTag::DYNAMIC_BOX; $crate::MAX_PARAMS],
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function that returns opaque pointer (raw pointer to opaque type, no params)
    ($name:expr, $func:ident, () -> opaque) => {{
        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: 0,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::DYNAMIC_BOX,  // Mark as opaque
            params: [$crate::TypeTag::VOID; $crate::MAX_PARAMS],
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns opaque pointer
    // Match: (...) -> opaque where ... is anything
    ($name:expr, $func:ident, ($($param:tt)*) -> opaque) => {{
        // Count and build param array
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::DYNAMIC_BOX,  // Mark as opaque
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns void
    ($name:expr, $func:ident, ($($param:tt)*) -> void) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::VOID,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns f32
    ($name:expr, $func:ident, ($($param:tt)*) -> f32) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::F32,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns f64
    ($name:expr, $func:ident, ($($param:tt)*) -> f64) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::F64,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns i32
    ($name:expr, $func:ident, ($($param:tt)*) -> i32) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::I32,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns i64
    ($name:expr, $func:ident, ($($param:tt)*) -> i64) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::I64,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns u32
    ($name:expr, $func:ident, ($($param:tt)*) -> u32) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::U32,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns u64
    ($name:expr, $func:ident, ($($param:tt)*) -> u64) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::U64,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns bool
    ($name:expr, $func:ident, ($($param:tt)*) -> bool) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::BOOL,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Function with any params that returns u8
    ($name:expr, $func:ident, ($($param:tt)*) -> u8) => {{
        const PARAM_COUNT: u8 = $crate::__count_params!($($param)*);
        const PARAM_ARRAY: [$crate::TypeTag; $crate::MAX_PARAMS] = $crate::__build_param_array!($($param)*);

        static SIG: $crate::ZrtlSymbolSig = $crate::ZrtlSymbolSig {
            param_count: PARAM_COUNT,
            flags: $crate::ZrtlSigFlags::NONE,
            return_type: $crate::TypeTag::U8,
            params: PARAM_ARRAY,
        };
        $crate::ZrtlSymbol::with_sig(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char,
            $func as *const u8,
            &SIG as *const $crate::ZrtlSymbolSig,
        )
    }};
    // Legacy: no signature (backwards compatible)
    ($name:expr, $func:ident) => {
        $crate::zrtl_symbol!($name, $func)
    };
}

/// Internal helper to convert symbol entry to ZrtlSymbol
#[macro_export]
#[doc(hidden)]
macro_rules! __zrtl_symbol_entry {
    // All-dynamic signature (all params are DynamicBox)
    ($sym_name:expr, $func:ident, dynamic($count:expr) -> void) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, dynamic($count) -> void)
    };
    // All-dynamic signature with dynamic return
    ($sym_name:expr, $func:ident, dynamic($count:expr) -> dynamic) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, dynamic($count) -> dynamic)
    };
    // Opaque return type (no params)
    ($sym_name:expr, $func:ident, () -> opaque) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, () -> opaque)
    };
    // Opaque return type (with params)
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> opaque) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> opaque)
    };
    // Void return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> void) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> void)
    };
    // F32 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> f32) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> f32)
    };
    // F64 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> f64) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> f64)
    };
    // I32 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> i32) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> i32)
    };
    // I64 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> i64) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> i64)
    };
    // U32 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> u32) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> u32)
    };
    // U64 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> u64) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> u64)
    };
    // Bool return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> bool) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> bool)
    };
    // U8 return type
    ($sym_name:expr, $func:ident, ($($param:tt)*) -> u8) => {
        $crate::zrtl_symbol_sig!($sym_name, $func, ($($param)*) -> u8)
    };
    // No signature (backwards compatible)
    ($sym_name:expr, $func:ident) => {
        $crate::zrtl_symbol!($sym_name, $func)
    };
}

/// Internal helper to count symbol entries
#[macro_export]
#[doc(hidden)]
macro_rules! __count_symbols {
    () => { 0 };
    (($($tt:tt)*) $($rest:tt)*) => { 1 + $crate::__count_symbols!($($rest)*) };
}

/// Internal helper to count parameters in a signature
#[macro_export]
#[doc(hidden)]
macro_rules! __count_params {
    () => { 0 };
    (f32) => { 1 };
    (f64) => { 1 };
    (i32) => { 1 };
    (i64) => { 1 };
    (u8) => { 1 };
    (u32) => { 1 };
    (u64) => { 1 };
    (bool) => { 1 };
    (f32, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (f64, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (i32, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (i64, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (u8, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (u32, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (u64, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
    (bool, $($rest:tt)*) => { 1 + $crate::__count_params!($($rest)*) };
}

/// Internal helper to build parameter type array
#[macro_export]
#[doc(hidden)]
macro_rules! __build_param_array {
    () => { [$crate::TypeTag::VOID; $crate::MAX_PARAMS] };
    ($($param:tt)*) => {{
        let mut params = [$crate::TypeTag::VOID; $crate::MAX_PARAMS];
        $crate::__fill_param_array!(params, 0, $($param)*);
        params
    }};
}

/// Internal helper to fill parameter array
#[macro_export]
#[doc(hidden)]
macro_rules! __fill_param_array {
    ($array:ident, $idx:expr,) => {};
    ($array:ident, $idx:expr, f32) => {
        $array[$idx] = $crate::TypeTag::F32;
    };
    ($array:ident, $idx:expr, f64) => {
        $array[$idx] = $crate::TypeTag::F64;
    };
    ($array:ident, $idx:expr, i32) => {
        $array[$idx] = $crate::TypeTag::I32;
    };
    ($array:ident, $idx:expr, i64) => {
        $array[$idx] = $crate::TypeTag::I64;
    };
    ($array:ident, $idx:expr, u8) => {
        $array[$idx] = $crate::TypeTag::U8;
    };
    ($array:ident, $idx:expr, u32) => {
        $array[$idx] = $crate::TypeTag::U32;
    };
    ($array:ident, $idx:expr, u64) => {
        $array[$idx] = $crate::TypeTag::U64;
    };
    ($array:ident, $idx:expr, bool) => {
        $array[$idx] = $crate::TypeTag::BOOL;
    };
    ($array:ident, $idx:expr, f32, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::F32;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, f64, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::F64;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, i32, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::I32;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, i64, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::I64;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, u8, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::U8;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, u32, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::U32;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, u64, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::U64;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
    ($array:ident, $idx:expr, bool, $($rest:tt)*) => {
        $array[$idx] = $crate::TypeTag::BOOL;
        $crate::__fill_param_array!($array, $idx + 1, $($rest)*);
    };
}

/// Macro to define a complete ZRTL plugin
///
/// This creates both the symbol table and plugin info exports.
///
/// # Example (legacy, no signatures)
///
/// ```ignore
/// zrtl_plugin! {
///     name: "my_runtime",
///     symbols: [
///         ("$Math$add", math_add),
///         ("$Math$sub", math_sub),
///     ]
/// }
/// ```
///
/// # Example (with signatures for auto-boxing)
///
/// ```ignore
/// zrtl_plugin! {
///     name: "io_runtime",
///     symbols: [
///         ("$IO$print_str", io_print_str),                    // No signature (legacy)
///         ("$IO$print", io_print, dynamic(1) -> void),        // All params are DynamicBox
///         ("$IO$format", io_format, dynamic(1) -> dynamic),   // DynamicBox in and out
///     ]
/// }
/// ```
#[macro_export]
macro_rules! zrtl_plugin {
    // New unified syntax that supports both legacy and signature forms
    (name: $name:expr, symbols: [$( ($($entry:tt)*) ),* $(,)?]) => {
        // Plugin info export
        #[no_mangle]
        pub static _zrtl_info: $crate::ZrtlInfo = $crate::ZrtlInfo::new(
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char
        );

        // Symbol table export - use a static array with C ABI layout
        // The loader expects *const ZrtlSymbol (a simple pointer to the first element)
        #[no_mangle]
        pub static _zrtl_symbols: [$crate::ZrtlSymbol; {
            // Count symbols + 1 for sentinel
            $crate::__count_symbols!($(($($entry)*))*)  + 1
        }] = [
            $(
                $crate::__zrtl_symbol_entry!($($entry)*),
            )*
            $crate::ZrtlSymbol::null(), // Sentinel
        ];
    };
}

/// Trait for types that can be registered as ZRTL types
pub trait ZrtlTyped: Sized {
    /// Get the type name
    fn type_name() -> &'static str;

    /// Get the type size
    fn type_size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Get the type alignment
    fn type_alignment() -> usize {
        std::mem::align_of::<Self>()
    }

    /// Get the type category
    fn type_category() -> crate::TypeCategory;
}

/// Type info for registered custom types
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: &'static str,
    /// Size in bytes
    pub size: u32,
    /// Alignment requirement
    pub alignment: u32,
    /// Type category
    pub category: crate::TypeCategory,
}

impl TypeInfo {
    /// Create type info for a type implementing ZrtlTyped
    pub fn from_typed<T: ZrtlTyped>() -> Self {
        Self {
            name: T::type_name(),
            size: T::type_size() as u32,
            alignment: T::type_alignment() as u32,
            category: T::type_category(),
        }
    }
}

// Implement ZrtlTyped for primitive types
macro_rules! impl_zrtl_typed_primitive {
    ($ty:ty, $category:ident) => {
        impl ZrtlTyped for $ty {
            fn type_name() -> &'static str {
                stringify!($ty)
            }

            fn type_category() -> crate::TypeCategory {
                crate::TypeCategory::$category
            }
        }
    };
}

impl_zrtl_typed_primitive!(bool, Bool);
impl_zrtl_typed_primitive!(i8, Int);
impl_zrtl_typed_primitive!(i16, Int);
impl_zrtl_typed_primitive!(i32, Int);
impl_zrtl_typed_primitive!(i64, Int);
impl_zrtl_typed_primitive!(isize, Int);
impl_zrtl_typed_primitive!(u8, UInt);
impl_zrtl_typed_primitive!(u16, UInt);
impl_zrtl_typed_primitive!(u32, UInt);
impl_zrtl_typed_primitive!(u64, UInt);
impl_zrtl_typed_primitive!(usize, UInt);
impl_zrtl_typed_primitive!(f32, Float);
impl_zrtl_typed_primitive!(f64, Float);

// ============================================================================
// Test Framework Macros
// ============================================================================

/// ZRTL-specific assertion macro with enhanced error messages
///
/// Works like `assert!()` but provides better output for ZRTL tests.
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert;
///
/// let x = 5;
/// zrtl_assert!(x > 0);
/// zrtl_assert!(x > 0, "x must be positive, got {}", x);
/// ```
#[macro_export]
macro_rules! zrtl_assert {
    ($cond:expr) => {
        if !$cond {
            panic!(
                "[ZRTL] Assertion failed: `{}`\n  at {}:{}",
                stringify!($cond),
                file!(),
                line!()
            );
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            panic!(
                "[ZRTL] Assertion failed: `{}`\n  at {}:{}\n  {}",
                stringify!($cond),
                file!(),
                line!(),
                format_args!($($arg)+)
            );
        }
    };
}

/// ZRTL-specific equality assertion with enhanced error messages
///
/// Works like `assert_eq!()` but provides better output for ZRTL tests.
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_eq;
///
/// let arr_len = 3;
/// zrtl_assert_eq!(arr_len, 3);
/// zrtl_assert_eq!(arr_len, 3, "Array should have 3 elements");
/// ```
#[macro_export]
macro_rules! zrtl_assert_eq {
    ($left:expr, $right:expr) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    panic!(
                        "[ZRTL] Assertion failed: `{} == {}`\n  left:  {:?}\n  right: {:?}\n  at {}:{}",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        right_val,
                        file!(),
                        line!()
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    panic!(
                        "[ZRTL] Assertion failed: `{} == {}`\n  left:  {:?}\n  right: {:?}\n  at {}:{}\n  {}",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        right_val,
                        file!(),
                        line!(),
                        format_args!($($arg)+)
                    );
                }
            }
        }
    };
}

/// ZRTL-specific inequality assertion with enhanced error messages
///
/// Works like `assert_ne!()` but provides better output for ZRTL tests.
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_ne;
///
/// let x = 5;
/// let y = 10;
/// zrtl_assert_ne!(x, y);
/// zrtl_assert_ne!(x, y, "Values should be different");
/// ```
#[macro_export]
macro_rules! zrtl_assert_ne {
    ($left:expr, $right:expr) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    panic!(
                        "[ZRTL] Assertion failed: `{} != {}`\n  both equal: {:?}\n  at {}:{}",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        file!(),
                        line!()
                    );
                }
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    panic!(
                        "[ZRTL] Assertion failed: `{} != {}`\n  both equal: {:?}\n  at {}:{}\n  {}",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        file!(),
                        line!(),
                        format_args!($($arg)+)
                    );
                }
            }
        }
    };
}

/// ZRTL assertion for Option types - asserts that the value is Some
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_some;
///
/// let value: Option<i32> = Some(42);
/// let result = zrtl_assert_some!(value);
/// assert_eq!(result, 42);
/// ```
#[macro_export]
macro_rules! zrtl_assert_some {
    ($opt:expr) => {
        match $opt {
            Some(v) => v,
            None => panic!(
                "[ZRTL] Assertion failed: expected Some, got None\n  expression: `{}`\n  at {}:{}",
                stringify!($opt),
                file!(),
                line!()
            ),
        }
    };
    ($opt:expr, $($arg:tt)+) => {
        match $opt {
            Some(v) => v,
            None => panic!(
                "[ZRTL] Assertion failed: expected Some, got None\n  expression: `{}`\n  at {}:{}\n  {}",
                stringify!($opt),
                file!(),
                line!(),
                format_args!($($arg)+)
            ),
        }
    };
}

/// ZRTL assertion for Option types - asserts that the value is None
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_none;
///
/// let value: Option<i32> = None;
/// zrtl_assert_none!(value);
/// ```
#[macro_export]
macro_rules! zrtl_assert_none {
    ($opt:expr) => {
        if let Some(v) = &$opt {
            panic!(
                "[ZRTL] Assertion failed: expected None, got Some({:?})\n  expression: `{}`\n  at {}:{}",
                v,
                stringify!($opt),
                file!(),
                line!()
            );
        }
    };
    ($opt:expr, $($arg:tt)+) => {
        if let Some(v) = &$opt {
            panic!(
                "[ZRTL] Assertion failed: expected None, got Some({:?})\n  expression: `{}`\n  at {}:{}\n  {}",
                v,
                stringify!($opt),
                file!(),
                line!(),
                format_args!($($arg)+)
            );
        }
    };
}

/// ZRTL assertion for Result types - asserts that the value is Ok
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_ok;
///
/// let result: Result<i32, &str> = Ok(42);
/// let value = zrtl_assert_ok!(result);
/// assert_eq!(value, 42);
/// ```
#[macro_export]
macro_rules! zrtl_assert_ok {
    ($result:expr) => {
        match $result {
            Ok(v) => v,
            Err(e) => panic!(
                "[ZRTL] Assertion failed: expected Ok, got Err({:?})\n  expression: `{}`\n  at {}:{}",
                e,
                stringify!($result),
                file!(),
                line!()
            ),
        }
    };
    ($result:expr, $($arg:tt)+) => {
        match $result {
            Ok(v) => v,
            Err(e) => panic!(
                "[ZRTL] Assertion failed: expected Ok, got Err({:?})\n  expression: `{}`\n  at {}:{}\n  {}",
                e,
                stringify!($result),
                file!(),
                line!(),
                format_args!($($arg)+)
            ),
        }
    };
}

/// ZRTL assertion for Result types - asserts that the value is Err
///
/// # Example
///
/// ```rust
/// use zrtl::zrtl_assert_err;
///
/// let result: Result<i32, &str> = Err("error");
/// let error = zrtl_assert_err!(result);
/// assert_eq!(error, "error");
/// ```
#[macro_export]
macro_rules! zrtl_assert_err {
    ($result:expr) => {
        match $result {
            Err(e) => e,
            Ok(v) => panic!(
                "[ZRTL] Assertion failed: expected Err, got Ok({:?})\n  expression: `{}`\n  at {}:{}",
                v,
                stringify!($result),
                file!(),
                line!()
            ),
        }
    };
    ($result:expr, $($arg:tt)+) => {
        match $result {
            Err(e) => e,
            Ok(v) => panic!(
                "[ZRTL] Assertion failed: expected Err, got Ok({:?})\n  expression: `{}`\n  at {}:{}\n  {}",
                v,
                stringify!($result),
                file!(),
                line!(),
                format_args!($($arg)+)
            ),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_layout() {
        // Verify C ABI compatibility
        // ZrtlSymbol now has 3 pointer-sized fields: name, ptr, sig
        assert_eq!(
            std::mem::size_of::<ZrtlSymbol>(),
            std::mem::size_of::<*const u8>() * 3
        );
    }

    #[test]
    fn test_info_layout() {
        // Verify C ABI compatibility - should be 16 bytes on 64-bit (4 byte version + padding + 8 byte pointer)
        assert!(std::mem::size_of::<ZrtlInfo>() >= 12);
    }

    #[test]
    fn test_type_info() {
        let info = TypeInfo::from_typed::<i32>();
        assert_eq!(info.name, "i32");
        assert_eq!(info.size, 4);
        assert_eq!(info.category, crate::TypeCategory::Int);
    }

    #[test]
    fn test_zrtl_assert() {
        zrtl_assert!(true);
        zrtl_assert!(1 + 1 == 2);
        zrtl_assert!(1 < 2, "one should be less than two");
    }

    #[test]
    fn test_zrtl_assert_eq() {
        zrtl_assert_eq!(1, 1);
        zrtl_assert_eq!("hello", "hello");
        zrtl_assert_eq!(vec![1, 2, 3], vec![1, 2, 3], "vectors should match");
    }

    #[test]
    fn test_zrtl_assert_ne() {
        zrtl_assert_ne!(1, 2);
        zrtl_assert_ne!("hello", "world");
    }

    #[test]
    fn test_zrtl_assert_some() {
        let opt: Option<i32> = Some(42);
        let value = zrtl_assert_some!(opt);
        assert_eq!(value, 42);
    }

    #[test]
    fn test_zrtl_assert_none() {
        let opt: Option<i32> = None;
        zrtl_assert_none!(opt);
    }

    #[test]
    fn test_zrtl_assert_ok() {
        let result: Result<i32, &str> = Ok(42);
        let value = zrtl_assert_ok!(result);
        assert_eq!(value, 42);
    }

    #[test]
    fn test_zrtl_assert_err() {
        let result: Result<i32, &str> = Err("error");
        let error = zrtl_assert_err!(result);
        assert_eq!(error, "error");
    }
}
