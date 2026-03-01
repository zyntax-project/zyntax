#![allow(unused, dead_code, deprecated)]

//! Haxe-specific Runtime for Zyntax
//!
//! This crate provides Haxe-specific runtime functions for the Zyntax compiler.
//! It's separate from the generic zyntax_runtime to keep Haxe-specific implementations
//! localized and easier to maintain.
//!
//! # Functions
//!
//! - `trace` - Haxe's standard debug output function
//! - String operations specific to Haxe's string representation
//! - Dynamic type handling for Haxe's Dynamic type

extern crate libc;

use zyntax_compiler::zrtl::{DynamicValue, TypeCategory, TypeId};
use zyntax_plugin_macros::{runtime_export, runtime_plugin};

// Declare this as the Haxe runtime plugin
runtime_plugin! {
    name: "haxe",
}

// ============================================================================
// String Header Constants (matching reflaxe.zyntax string format)
// ============================================================================

/// Header size for length-prefixed Haxe strings (1 i32 = 4 bytes)
const HEADER_SIZE: isize = 1;

// ============================================================================
// Haxe trace() Function
// ============================================================================

/// Haxe trace function - prints a string value with source location
/// This is the standard Haxe debugging output function.
///
/// In Haxe: trace("Hello") outputs: "Test.hx:10: Hello"
///
/// Parameters:
/// - value: The string to print (length-prefixed Haxe string)
/// - pos_info: Optional position info string (e.g., "Test.hx:10:")
#[runtime_export("$haxe$trace")]
pub unsafe extern "C" fn haxe_trace(value: *const i32, pos_info: *const i32) {
    unsafe {
        // Print position info if provided
        if !pos_info.is_null() {
            let pos_len = *pos_info;
            let pos_chars = pos_info.offset(HEADER_SIZE) as *const u8;

            for i in 0..pos_len {
                let ch = *pos_chars.offset(i as isize);
                libc::putchar(ch as i32);
            }
            // Add space after position info
            libc::putchar(b' ' as i32);
        }

        // Print the value
        if !value.is_null() {
            let length = *value;
            let chars = value.offset(HEADER_SIZE) as *const u8;

            for i in 0..length {
                let ch = *chars.offset(i as isize);
                libc::putchar(ch as i32);
            }
        } else {
            // Print "null" for null values
            libc::printf(b"null\0".as_ptr() as *const i8);
        }

        // Always end with newline
        libc::putchar(b'\n' as i32);
    }
}

/// Simple trace function - just prints the string with newline
/// Used when no position info is available
#[runtime_export("$haxe$trace$simple")]
pub unsafe extern "C" fn haxe_trace_simple(value: *const i32) {
    haxe_trace(value, core::ptr::null());
}

/// Trace an integer value directly
#[runtime_export("$haxe$trace$int")]
pub unsafe extern "C" fn haxe_trace_int(value: i32, pos_info: *const i32) {
    unsafe {
        print_pos_info(pos_info);
        libc::printf(b"%d\n\0".as_ptr() as *const i8, value);
    }
}

/// Trace a float value directly
#[runtime_export("$haxe$trace$float")]
pub unsafe extern "C" fn haxe_trace_float(value: f64, pos_info: *const i32) {
    unsafe {
        print_pos_info(pos_info);
        libc::printf(b"%g\n\0".as_ptr() as *const i8, value);
    }
}

/// Trace a boolean value directly
#[runtime_export("$haxe$trace$bool")]
pub unsafe extern "C" fn haxe_trace_bool(value: i32, pos_info: *const i32) {
    unsafe {
        print_pos_info(pos_info);
        if value != 0 {
            libc::printf(b"true\n\0".as_ptr() as *const i8);
        } else {
            libc::printf(b"false\n\0".as_ptr() as *const i8);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Print position info helper (for internal use)
unsafe fn print_pos_info(pos_info: *const i32) {
    if !pos_info.is_null() {
        let pos_len = *pos_info;
        let pos_chars = pos_info.offset(HEADER_SIZE) as *const u8;

        for i in 0..pos_len {
            let ch = *pos_chars.offset(i as isize);
            libc::putchar(ch as i32);
        }
        // Add space after position info
        libc::putchar(b' ' as i32);
    }
}

/// Print a length-prefixed Haxe string (internal helper)
unsafe fn print_haxe_string(str_ptr: *const i32) {
    if !str_ptr.is_null() {
        let length = *str_ptr;
        let chars = str_ptr.offset(HEADER_SIZE) as *const u8;

        for i in 0..length {
            let ch = *chars.offset(i as isize);
            libc::putchar(ch as i32);
        }
    } else {
        libc::printf(b"null\0".as_ptr() as *const i8);
    }
}

// ============================================================================
// Dynamic Value trace support
// ============================================================================

/// Haxe String representation for Dynamic type (ptr, len, cap style)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeString {
    pub ptr: *mut u8, // Pointer to string data (UTF-8)
    pub len: usize,   // Length in bytes
    pub cap: usize,   // Capacity in bytes
}

/// Trace any Dynamic value - dispatches based on runtime type
///
/// This is the main trace entry point for Haxe's Dynamic type.
/// It inspects the DynamicValue's type tag and prints accordingly.
///
/// Parameters:
/// - dynamic_ptr: Pointer to a DynamicValue struct
/// - pos_info: Optional position info string (length-prefixed Haxe string)
#[runtime_export("$haxe$trace$any")]
pub unsafe extern "C" fn haxe_trace_any(dynamic_ptr: *const DynamicValue, pos_info: *const i32) {
    unsafe {
        print_pos_info(pos_info);

        if dynamic_ptr.is_null() {
            libc::printf(b"null\n\0".as_ptr() as *const i8);
            return;
        }

        let dynamic = &*dynamic_ptr;

        // Check if the value itself is null
        if dynamic.is_null() {
            libc::printf(b"null\n\0".as_ptr() as *const i8);
            return;
        }

        let type_id = dynamic.type_id();

        // Dispatch based on type category
        match type_id.category() {
            TypeCategory::Void => {
                libc::printf(b"null\n\0".as_ptr() as *const i8);
            }
            TypeCategory::Bool => {
                if let Some(&value) = dynamic.as_ref::<i32>() {
                    if value != 0 {
                        libc::printf(b"true\n\0".as_ptr() as *const i8);
                    } else {
                        libc::printf(b"false\n\0".as_ptr() as *const i8);
                    }
                } else {
                    libc::printf(b"<Bool?>\n\0".as_ptr() as *const i8);
                }
            }
            TypeCategory::Int => {
                // Handle different int sizes
                match type_id {
                    t if t == TypeId::I8 => {
                        if let Some(&v) = dynamic.as_ref::<i8>() {
                            libc::printf(b"%d\n\0".as_ptr() as *const i8, v as i32);
                        }
                    }
                    t if t == TypeId::I16 => {
                        if let Some(&v) = dynamic.as_ref::<i16>() {
                            libc::printf(b"%d\n\0".as_ptr() as *const i8, v as i32);
                        }
                    }
                    t if t == TypeId::I32 => {
                        if let Some(&v) = dynamic.as_ref::<i32>() {
                            libc::printf(b"%d\n\0".as_ptr() as *const i8, v);
                        }
                    }
                    _ => {
                        // I64 or ISIZE
                        if let Some(&v) = dynamic.as_ref::<i64>() {
                            libc::printf(b"%lld\n\0".as_ptr() as *const i8, v);
                        }
                    }
                }
            }
            TypeCategory::UInt => match type_id {
                t if t == TypeId::U8 => {
                    if let Some(&v) = dynamic.as_ref::<u8>() {
                        libc::printf(b"%u\n\0".as_ptr() as *const i8, v as u32);
                    }
                }
                t if t == TypeId::U16 => {
                    if let Some(&v) = dynamic.as_ref::<u16>() {
                        libc::printf(b"%u\n\0".as_ptr() as *const i8, v as u32);
                    }
                }
                t if t == TypeId::U32 => {
                    if let Some(&v) = dynamic.as_ref::<u32>() {
                        libc::printf(b"%u\n\0".as_ptr() as *const i8, v);
                    }
                }
                _ => {
                    if let Some(&v) = dynamic.as_ref::<u64>() {
                        libc::printf(b"%llu\n\0".as_ptr() as *const i8, v);
                    }
                }
            },
            TypeCategory::Float => {
                if type_id == TypeId::F32 {
                    if let Some(&v) = dynamic.as_ref::<f32>() {
                        libc::printf(b"%g\n\0".as_ptr() as *const i8, v as f64);
                    }
                } else {
                    if let Some(&v) = dynamic.as_ref::<f64>() {
                        libc::printf(b"%g\n\0".as_ptr() as *const i8, v);
                    }
                }
            }
            TypeCategory::String => {
                // String can be stored as either:
                // 1. A length-prefixed Haxe string (*const i32)
                // 2. A HaxeString struct (ptr, len, cap)

                // Try as length-prefixed string pointer first
                if let Some(&str_ptr) = dynamic.as_ref::<*const i32>() {
                    print_haxe_string(str_ptr);
                    libc::putchar(b'\n' as i32);
                } else if let Some(haxe_str) = dynamic.as_ref::<HaxeString>() {
                    // HaxeString struct (ptr, len, cap)
                    if !haxe_str.ptr.is_null() && haxe_str.len > 0 {
                        let slice = std::slice::from_raw_parts(haxe_str.ptr, haxe_str.len);
                        for &ch in slice {
                            libc::putchar(ch as i32);
                        }
                        libc::putchar(b'\n' as i32);
                    } else {
                        libc::printf(b"\"\"\n\0".as_ptr() as *const i8);
                    }
                } else {
                    libc::printf(b"<String?>\n\0".as_ptr() as *const i8);
                }
            }
            TypeCategory::Array => {
                // TODO: Implement array printing with element traversal
                libc::printf(b"[Array]\n\0".as_ptr() as *const i8);
            }
            TypeCategory::Struct | TypeCategory::Class => {
                // For struct/class, show address
                libc::printf(b"<Object@%p>\n\0".as_ptr() as *const i8, dynamic.value_ptr);
            }
            TypeCategory::Enum => {
                libc::printf(b"[Enum]\n\0".as_ptr() as *const i8);
            }
            TypeCategory::Function => {
                libc::printf(b"<function>\n\0".as_ptr() as *const i8);
            }
            TypeCategory::Optional => {
                libc::printf(b"[Optional]\n\0".as_ptr() as *const i8);
            }
            TypeCategory::Map => {
                libc::printf(b"[Map]\n\0".as_ptr() as *const i8);
            }
            _ => {
                // Unknown type - print address
                libc::printf(b"<Dynamic@%p>\n\0".as_ptr() as *const i8, dynamic.value_ptr);
            }
        }
    }
}

/// Trace any Dynamic value without position info
#[runtime_export("$haxe$trace$any$simple")]
pub unsafe extern "C" fn haxe_trace_any_simple(dynamic_ptr: *const DynamicValue) {
    haxe_trace_any(dynamic_ptr, core::ptr::null());
}

// ============================================================================
// ZRTL Plugin Export (for dynamic loading as .zrtl)
// ============================================================================
//
// These exports allow this library to be loaded dynamically as a ZRTL plugin.
// The `_zrtl_info` and `_zrtl_symbols` are required by the ZrtlPlugin loader.

use std::ffi::c_char;
use zyntax_compiler::zrtl::{ZrtlInfo, ZrtlSymbol, ZRTL_VERSION};

/// Plugin info - required for ZRTL dynamic loading
#[no_mangle]
pub static _zrtl_info: ZrtlInfo = ZrtlInfo {
    version: ZRTL_VERSION,
    name: b"haxe\0".as_ptr() as *const c_char,
};

/// Symbol table - required for ZRTL dynamic loading
/// Each symbol is { name, function_ptr } with null sentinel at end
#[no_mangle]
pub static _zrtl_symbols: [ZrtlSymbol; 8] = [
    ZrtlSymbol {
        name: b"$haxe$trace\0".as_ptr() as *const c_char,
        ptr: haxe_trace as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$simple\0".as_ptr() as *const c_char,
        ptr: haxe_trace_simple as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$int\0".as_ptr() as *const c_char,
        ptr: haxe_trace_int as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$float\0".as_ptr() as *const c_char,
        ptr: haxe_trace_float as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$bool\0".as_ptr() as *const c_char,
        ptr: haxe_trace_bool as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$any\0".as_ptr() as *const c_char,
        ptr: haxe_trace_any as *const u8,
        sig: std::ptr::null(),
    },
    ZrtlSymbol {
        name: b"$haxe$trace$any$simple\0".as_ptr() as *const c_char,
        ptr: haxe_trace_any_simple as *const u8,
        sig: std::ptr::null(),
    },
    // Sentinel - null name indicates end of symbol table
    ZrtlSymbol {
        name: std::ptr::null(),
        ptr: std::ptr::null(),
        sig: std::ptr::null(),
    },
];
