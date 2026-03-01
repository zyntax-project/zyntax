#![allow(unused, dead_code, deprecated)]

//! Zyntax Runtime Library
//!
//! This library provides **language-agnostic** runtime primitives that can be used
//! by any language frontend targeting Zyntax.
//!
//! Language-specific runtime implementations (like Haxe's Array, String, etc.)
//! should go in their respective frontend directories (e.g., reflaxe.zyntax/runtime/).
//!
//! All exported functions use C calling convention for compatibility with JIT-compiled code.
//!
//! # Plugin Architecture
//!
//! This runtime uses Zyntax's plugin system to automatically register standard I/O functions.

extern crate libc;

use zyntax_plugin_macros::{runtime_export, runtime_plugin};

// Declare this as the standard library runtime plugin
runtime_plugin! {
    name: "stdlib",
}

// ============================================================================
// I/O Runtime Functions
// ============================================================================

/// Print an i32 integer to stdout
///
/// # Safety
/// This function is marked unsafe because it's called from JIT code.
/// It's actually safe as it just prints to stdout via libc.
#[runtime_export("print_i32")]
pub extern "C" fn print_i32(value: i32) {
    // Use libc printf for now - later we can make this more sophisticated
    unsafe {
        let format_str = b"%d\0".as_ptr() as *const i8;
        libc::printf(format_str, value);
        libc::fflush(core::ptr::null_mut());
    }
}

/// Print an i64 integer to stdout
#[runtime_export("print_i64")]
pub extern "C" fn print_i64(value: i64) {
    unsafe {
        let format_str = b"%lld\0".as_ptr() as *const i8;
        libc::printf(format_str, value);
        libc::fflush(core::ptr::null_mut());
    }
}

/// Print a newline
#[runtime_export("print_newline")]
pub extern "C" fn print_newline() {
    unsafe {
        libc::putchar(b'\n' as i32);
        libc::fflush(core::ptr::null_mut());
    }
}

/// Print a C-style null-terminated string
#[runtime_export("print_cstr")]
pub extern "C" fn print_cstr(ptr: *const u8) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let format_str = b"%s\0".as_ptr() as *const i8;
        libc::printf(format_str, ptr as *const i8);
        libc::fflush(core::ptr::null_mut());
    }
}

/// Print a string with known length
#[runtime_export("print_str")]
pub unsafe extern "C" fn print_str(ptr: *const u8, len: i32) {
    if ptr.is_null() || len < 0 {
        return;
    }
    for i in 0..len {
        libc::putchar(*ptr.offset(i as isize) as i32);
    }
    libc::fflush(core::ptr::null_mut());
}

/// Print a string with newline
#[runtime_export("println_str")]
pub unsafe extern "C" fn println_str(ptr: *const u8, len: i32) {
    print_str(ptr, len);
    print_newline();
}

/// Print an integer with newline
#[runtime_export("println_i32")]
pub extern "C" fn println_i32(value: i32) {
    print_i32(value);
    print_newline();
}
