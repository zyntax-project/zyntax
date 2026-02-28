//! ZRTL I/O Plugin
//!
//! Provides standard I/O operations for Zyntax-based languages using
//! the ZRTL SDK string format `[i32 length][utf8_bytes...]`.
//!
//! ## Exported Symbols
//!
//! ### String Output
//! - `$IO$print` - Print ZRTL string to stdout (no newline)
//! - `$IO$println` - Print ZRTL string with newline to stdout
//! - `$IO$eprint` - Print ZRTL string to stderr (no newline)
//! - `$IO$eprintln` - Print ZRTL string with newline to stderr
//!
//! ### Primitive Output
//! - `$IO$print_i64`, `$IO$println_i64` - Print integers
//! - `$IO$print_f64`, `$IO$println_f64` - Print floats
//! - `$IO$print_bool`, `$IO$println_bool` - Print booleans
//!
//! ### Input
//! - `$IO$read_line` - Read line from stdin, returns ZRTL string
//! - `$IO$input` - Read line with prompt, returns ZRTL string
//!
//! ### Formatting
//! - `$IO$format_i64` - Format integer as ZRTL string
//! - `$IO$format_f64` - Format float as ZRTL string
//! - `$IO$format_bool` - Format boolean as ZRTL string

use std::io::{self, BufRead, Write};
use zrtl::{
    zrtl_plugin,
    StringConstPtr, StringPtr,
    string_length, string_data, string_new, string_free,
};

// ============================================================================
// DynamicBox Dropper Functions
// ============================================================================

/// Dropper for DynamicBox values whose `data` field is a ZRTL string pointer.
/// Frees the string using the ZRTL allocation layout (header + data bytes).
extern "C" fn drop_zrtl_string(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    unsafe { string_free(ptr as StringPtr) }
}


// ============================================================================
// String I/O Functions (using ZRTL string format)
// ============================================================================

/// Print a ZRTL string to stdout (no newline)
///
/// ZRTL string format: `[i32 length][utf8_bytes...]`
///
/// # Safety
/// The pointer must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn io_print(s: StringConstPtr) {
    if s.is_null() {
        return;
    }
    let len = string_length(s) as usize;
    let data = string_data(s);
    if len > 0 && !data.is_null() {
        let bytes = std::slice::from_raw_parts(data, len);
        if let Ok(str_slice) = std::str::from_utf8(bytes) {
            print!("{}", str_slice);
        }
    }
}

/// Print a ZRTL string to stdout with newline
///
/// # Safety
/// The pointer must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn io_println(s: StringConstPtr) {
    if s.is_null() {
        println!();
        return;
    }
    let len = string_length(s) as usize;
    let data = string_data(s);
    if len > 0 && !data.is_null() {
        let bytes = std::slice::from_raw_parts(data, len);
        if let Ok(str_slice) = std::str::from_utf8(bytes) {
            println!("{}", str_slice);
        } else {
            println!();
        }
    } else {
        println!();
    }
}

/// Print a ZRTL string to stderr (no newline)
///
/// # Safety
/// The pointer must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn io_eprint(s: StringConstPtr) {
    if s.is_null() {
        return;
    }
    let len = string_length(s) as usize;
    let data = string_data(s);
    if len > 0 && !data.is_null() {
        let bytes = std::slice::from_raw_parts(data, len);
        if let Ok(str_slice) = std::str::from_utf8(bytes) {
            eprint!("{}", str_slice);
        }
    }
}

/// Print a ZRTL string to stderr with newline
///
/// # Safety
/// The pointer must be a valid ZRTL string pointer.
#[no_mangle]
pub unsafe extern "C" fn io_eprintln(s: StringConstPtr) {
    if s.is_null() {
        eprintln!();
        return;
    }
    let len = string_length(s) as usize;
    let data = string_data(s);
    if len > 0 && !data.is_null() {
        let bytes = std::slice::from_raw_parts(data, len);
        if let Ok(str_slice) = std::str::from_utf8(bytes) {
            eprintln!("{}", str_slice);
        } else {
            eprintln!();
        }
    } else {
        eprintln!();
    }
}

// ============================================================================
// Primitive I/O Functions
// ============================================================================

/// Print an i64 to stdout
#[no_mangle]
pub extern "C" fn io_print_i64(value: i64) {
    print!("{}", value);
}

/// Print an i64 to stdout with newline
#[no_mangle]
pub extern "C" fn io_println_i64(value: i64) {
    println!("{}", value);
}

/// Print a u64 to stdout
#[no_mangle]
pub extern "C" fn io_print_u64(value: u64) {
    print!("{}", value);
}

/// Print a u64 to stdout with newline
#[no_mangle]
pub extern "C" fn io_println_u64(value: u64) {
    println!("{}", value);
}

/// Print an f64 to stdout
#[no_mangle]
pub extern "C" fn io_print_f64(value: f64) {
    print!("{}", value);
}

/// Print an f64 to stdout with newline
#[no_mangle]
pub extern "C" fn io_println_f64(value: f64) {
    println!("{}", value);
}

/// Print a boolean to stdout
#[no_mangle]
pub extern "C" fn io_print_bool(value: bool) {
    print!("{}", value);
}

/// Print a boolean to stdout with newline
#[no_mangle]
pub extern "C" fn io_println_bool(value: bool) {
    println!("{}", value);
}

/// Print a single character (Unicode codepoint) to stdout
#[no_mangle]
pub extern "C" fn io_print_char(codepoint: u32) {
    if let Some(c) = char::from_u32(codepoint) {
        print!("{}", c);
    }
}

/// Print a single character with newline
#[no_mangle]
pub extern "C" fn io_println_char(codepoint: u32) {
    if let Some(c) = char::from_u32(codepoint) {
        println!("{}", c);
    } else {
        println!();
    }
}

// ============================================================================
// Flush Functions
// ============================================================================

/// Flush stdout
/// Returns 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn io_flush() -> i32 {
    match io::stdout().flush() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Flush stderr
/// Returns 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn io_flush_stderr() -> i32 {
    match io::stderr().flush() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ============================================================================
// Input Functions (returning ZRTL strings)
// ============================================================================

/// Read a line from stdin
///
/// Returns a ZRTL string pointer `[i32 length][utf8_bytes...]`
/// The caller must free this memory using `string_free` from the SDK.
/// Returns null on EOF or error.
#[no_mangle]
pub extern "C" fn io_read_line() -> StringPtr {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut buffer = String::new();

    match handle.read_line(&mut buffer) {
        Ok(0) => std::ptr::null_mut(), // EOF
        Ok(_) => {
            // Remove trailing newline
            if buffer.ends_with('\n') {
                buffer.pop();
                if buffer.ends_with('\r') {
                    buffer.pop();
                }
            }
            // Return as ZRTL string
            string_new(&buffer)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Read a line from stdin with a prompt
///
/// The prompt is a ZRTL string that will be printed before reading.
/// Returns a ZRTL string pointer, caller must free with `string_free`.
///
/// # Safety
/// The prompt pointer must be a valid ZRTL string pointer or null.
#[no_mangle]
pub unsafe extern "C" fn io_input(prompt: StringConstPtr) -> StringPtr {
    // Print prompt if provided
    if !prompt.is_null() {
        io_print(prompt);
        let _ = io::stdout().flush();
    }
    io_read_line()
}

// ============================================================================
// Formatting Functions (returning ZRTL strings)
// ============================================================================

/// Format an i64 as a ZRTL string
///
/// Returns a ZRTL string pointer, caller must free with `string_free`.
#[no_mangle]
pub extern "C" fn io_format_i64(value: i64) -> StringPtr {
    let formatted = format!("{}", value);
    string_new(&formatted)
}

/// Format a u64 as a ZRTL string
#[no_mangle]
pub extern "C" fn io_format_u64(value: u64) -> StringPtr {
    let formatted = format!("{}", value);
    string_new(&formatted)
}

/// Format an f64 as a ZRTL string
#[no_mangle]
pub extern "C" fn io_format_f64(value: f64) -> StringPtr {
    let formatted = format!("{}", value);
    string_new(&formatted)
}

/// Format an f64 with specified decimal precision
#[no_mangle]
pub extern "C" fn io_format_f64_precision(value: f64, precision: u32) -> StringPtr {
    let formatted = format!("{:.prec$}", value, prec = precision as usize);
    string_new(&formatted)
}

/// Format a boolean as a ZRTL string ("true" or "false")
#[no_mangle]
pub extern "C" fn io_format_bool(value: bool) -> StringPtr {
    string_new(if value { "true" } else { "false" })
}

/// Format an i64 in hexadecimal (lowercase)
#[no_mangle]
pub extern "C" fn io_format_hex(value: i64) -> StringPtr {
    let formatted = format!("{:x}", value);
    string_new(&formatted)
}

/// Format an i64 in hexadecimal (uppercase)
#[no_mangle]
pub extern "C" fn io_format_hex_upper(value: i64) -> StringPtr {
    let formatted = format!("{:X}", value);
    string_new(&formatted)
}

/// Format an i64 in binary
#[no_mangle]
pub extern "C" fn io_format_binary(value: i64) -> StringPtr {
    let formatted = format!("{:b}", value);
    string_new(&formatted)
}

/// Format an i64 in octal
#[no_mangle]
pub extern "C" fn io_format_octal(value: i64) -> StringPtr {
    let formatted = format!("{:o}", value);
    string_new(&formatted)
}

/// Format a Unicode codepoint as a ZRTL string
#[no_mangle]
pub extern "C" fn io_format_char(codepoint: u32) -> StringPtr {
    if let Some(c) = char::from_u32(codepoint) {
        let formatted = format!("{}", c);
        string_new(&formatted)
    } else {
        string_new("")
    }
}

// ============================================================================
// String Memory Management (re-exported from SDK)
// ============================================================================

/// Free a ZRTL string allocated by this module
///
/// # Safety
/// The pointer must have been allocated by io_read_line, io_input, or io_format_*
#[no_mangle]
pub unsafe extern "C" fn io_string_free(s: StringPtr) {
    string_free(s);
}

// ============================================================================
// Dynamic Print (DynamicBox) - Handles ALL known ZRTL types
// ============================================================================

use std::fmt::Write as FmtWrite;

/// Format a DynamicBox value to a String (internal helper)
///
/// This handles ALL known ZRTL runtime types including:
/// - Primitives: Void, Bool, Int, UInt, Float
/// - String: ZRTL string format [i32 length][utf8 bytes]
/// - Array: ZRTL array format [i32 cap][i32 len][elements]
/// - Tuple: Sequence of DynamicBox values
/// - Optional: None or Some(value)
/// - Pointer: Raw pointer display
/// - Struct/Class: Named struct with fields
/// - Enum: Variant index + optional payload
/// - Function: Function pointer
/// - Map: Key-value pairs (simplified display)
/// - TraitObject: Trait object (vtable + data)
/// - Opaque/Custom: Hex dump for unknown types
unsafe fn format_dynamic_box(value: &zrtl::DynamicBox, output: &mut String) {
    use zrtl::TypeCategory;

    if value.is_null() {
        output.push_str("null");
        return;
    }

    match value.category() {
        TypeCategory::Void => {
            output.push_str("void");
        }

        TypeCategory::Bool => {
            if let Some(b) = value.as_ref::<bool>() {
                let _ = write!(output, "{}", b);
            } else if let Some(b) = value.as_ref::<u8>() {
                let _ = write!(output, "{}", *b != 0);
            }
        }

        TypeCategory::Int => {
            match value.size {
                1 => { if let Some(v) = value.as_ref::<i8>() { let _ = write!(output, "{}", v); } }
                2 => { if let Some(v) = value.as_ref::<i16>() { let _ = write!(output, "{}", v); } }
                4 => { if let Some(v) = value.as_ref::<i32>() { let _ = write!(output, "{}", v); } }
                8 => { if let Some(v) = value.as_ref::<i64>() { let _ = write!(output, "{}", v); } }
                16 => { if let Some(v) = value.as_ref::<i128>() { let _ = write!(output, "{}", v); } }
                _ => { let _ = write!(output, "<int{}>", value.size * 8); }
            }
        }

        TypeCategory::UInt => {
            match value.size {
                1 => { if let Some(v) = value.as_ref::<u8>() { let _ = write!(output, "{}", v); } }
                2 => { if let Some(v) = value.as_ref::<u16>() { let _ = write!(output, "{}", v); } }
                4 => { if let Some(v) = value.as_ref::<u32>() { let _ = write!(output, "{}", v); } }
                8 => { if let Some(v) = value.as_ref::<u64>() { let _ = write!(output, "{}", v); } }
                16 => { if let Some(v) = value.as_ref::<u128>() { let _ = write!(output, "{}", v); } }
                _ => { let _ = write!(output, "<uint{}>", value.size * 8); }
            }
        }

        TypeCategory::Float => {
            match value.size {
                4 => { if let Some(v) = value.as_ref::<f32>() { let _ = write!(output, "{}", v); } }
                8 => { if let Some(v) = value.as_ref::<f64>() { let _ = write!(output, "{}", v); } }
                _ => { let _ = write!(output, "<float{}>", value.size * 8); }
            }
        }

        TypeCategory::String => {
            // ZRTL string format: [i32 length][utf8 bytes]
            let ptr = value.data as StringConstPtr;
            if !ptr.is_null() {
                let len = string_length(ptr) as usize;
                let data = string_data(ptr);
                if len > 0 && !data.is_null() {
                    let bytes = std::slice::from_raw_parts(data, len);
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        output.push_str(s);
                    } else {
                        let _ = write!(output, "<invalid utf8, {} bytes>", len);
                    }
                }
            }
        }

        TypeCategory::Array => {
            // ZRTL array format: [i32 capacity][i32 length][elements...]
            let arr_ptr = value.data as zrtl::ArrayConstPtr;
            if arr_ptr.is_null() {
                output.push_str("[]");
                return;
            }

            let capacity = zrtl::array_capacity(arr_ptr) as usize;
            let len = zrtl::array_length(arr_ptr) as usize;

            // Validate: length must not exceed capacity
            if len > capacity || capacity > (1 << 28) {
                let _ = write!(output, "<array len={} cap={} (invalid)>", len, capacity);
                return;
            }

            output.push('[');

            // Determine element type from size (heuristic)
            // For now, assume elements are same size as (total_size - 8) / len
            let elem_size = if len > 0 && value.size > 8 {
                ((value.size as usize) - 8) / len
            } else {
                4 // Default to 4 bytes (i32/f32)
            };

            // Try to print elements based on common element sizes
            let data_ptr = zrtl::array_data::<u8>(arr_ptr);
            if data_ptr.is_null() {
                output.push_str("<null data>");
                output.push(']');
                return;
            }

            let display_len = len.min(10);
            for i in 0..display_len {
                if i > 0 { output.push_str(", "); }

                match elem_size {
                    4 => {
                        // Could be i32 or f32, try f32 first for tensor arrays
                        let elem = *(data_ptr.add(i * 4) as *const f32);
                        if elem.is_finite() && elem.abs() < 1e10 {
                            let _ = write!(output, "{}", elem);
                        } else {
                            let elem_i = *(data_ptr.add(i * 4) as *const i32);
                            let _ = write!(output, "{}", elem_i);
                        }
                    }
                    8 => {
                        let elem = *(data_ptr.add(i * 8) as *const f64);
                        if elem.is_finite() && elem.abs() < 1e15 {
                            let _ = write!(output, "{}", elem);
                        } else {
                            let elem_i = *(data_ptr.add(i * 8) as *const i64);
                            let _ = write!(output, "{}", elem_i);
                        }
                    }
                    1 => {
                        let elem = *data_ptr.add(i);
                        let _ = write!(output, "{}", elem);
                    }
                    2 => {
                        let elem = *(data_ptr.add(i * 2) as *const i16);
                        let _ = write!(output, "{}", elem);
                    }
                    _ => {
                        let _ = write!(output, "?");
                    }
                }
            }

            if len > 10 {
                let _ = write!(output, ", ... ({} more)", len - 10);
            }
            output.push(']');
        }

        TypeCategory::Tuple => {
            // Tuple is stored as consecutive DynamicBox values
            // Size tells us total bytes, each DynamicBox is 32 bytes (on 64-bit)
            let box_size = std::mem::size_of::<zrtl::DynamicBox>();
            if box_size == 0 || (value.size as usize) % box_size != 0 {
                let _ = write!(output, "<tuple {} bytes (misaligned)>", value.size);
                return;
            }
            let num_elements = value.size as usize / box_size;
            if num_elements > 256 {
                let _ = write!(output, "<tuple {} elements (suspicious)>", num_elements);
                return;
            }
            output.push('(');
            let elements = value.data as *const zrtl::DynamicBox;
            if elements.is_null() {
                output.push_str("<null>)");
                return;
            }
            let display_count = num_elements.min(10);
            for i in 0..display_count {
                if i > 0 { output.push_str(", "); }
                let elem = &*elements.add(i);
                format_dynamic_box(elem, output);
            }
            if num_elements > 10 {
                let _ = write!(output, ", ... ({} more)", num_elements - 10);
            }
            output.push(')');
        }

        TypeCategory::Optional => {
            // Optional: first byte is 0 (None) or 1 (Some), followed by value
            if value.size == 0 || value.data.is_null() {
                output.push_str("None");
            } else {
                let has_value = *value.data;
                if has_value == 0 {
                    output.push_str("None");
                } else {
                    output.push_str("Some(");
                    // The actual value follows the discriminant
                    // This is a simplified representation
                    let _ = write!(output, "<{} bytes>", value.size - 1);
                    output.push(')');
                }
            }
        }

        TypeCategory::Result => {
            // Result: first byte is 0 (Ok) or 1 (Err), followed by value
            if value.size == 0 || value.data.is_null() {
                output.push_str("Result(?)");
            } else {
                let is_err = *value.data;
                if is_err == 0 {
                    output.push_str("Ok(...)");
                } else {
                    output.push_str("Err(...)");
                }
            }
        }

        TypeCategory::Pointer => {
            let ptr_val = if value.size == 8 {
                value.as_ref::<u64>().map(|p| *p as usize).unwrap_or(0)
            } else {
                value.as_ref::<u32>().map(|p| *p as usize).unwrap_or(0)
            };
            let _ = write!(output, "0x{:x}", ptr_val);
        }

        TypeCategory::Function => {
            let _ = write!(output, "<fn@{:p}>", value.data);
        }

        TypeCategory::Struct | TypeCategory::Class => {
            // For struct/class, we don't have field names, just show a generic representation
            let _ = write!(output, "<struct {} bytes@{:p}>", value.size, value.data);
        }

        TypeCategory::Enum => {
            // Enum: typically a tag byte followed by payload
            if value.size > 0 && !value.data.is_null() {
                let tag = *value.data;
                let _ = write!(output, "<enum variant {}@{:p}>", tag, value.data);
            } else {
                output.push_str("<enum>");
            }
        }

        TypeCategory::Union => {
            let _ = write!(output, "<union {} bytes@{:p}>", value.size, value.data);
        }

        TypeCategory::Map => {
            let _ = write!(output, "<map@{:p}>", value.data);
        }

        TypeCategory::TraitObject => {
            let _ = write!(output, "<dyn trait@{:p}>", value.data);
        }

        TypeCategory::Opaque | TypeCategory::Custom => {
            // Check if this opaque type has a Display trait implementation
            if let Some(display_fn) = value.display_fn {
                // Call the display function to format the value
                let formatted_ptr = display_fn(value.data as *const u8);
                if !formatted_ptr.is_null() {
                    // Display function returned a heap-allocated ZRTL string pointer
                    let str_ptr = formatted_ptr as StringConstPtr;
                    if let Some(formatted_str) = zrtl::string_as_str(str_ptr) {
                        output.push_str(formatted_str);
                    } else {
                        output.push_str("<opaque (display invalid utf8)>");
                    }
                    // Free the display function's returned string (it was heap-allocated)
                    string_free(formatted_ptr as StringPtr);
                } else {
                    // Display function returned null - fall back to hex dump
                    output.push_str("<opaque (display failed)>");
                }
            } else {
                // No Display trait - do NOT dereference opaque data pointers.
                // Some runtime-backed values point to foreign allocations that are
                // not safely readable as raw bytes from this plugin.
                let kind = if value.category() == TypeCategory::Custom {
                    "custom"
                } else {
                    "opaque"
                };
                let type_id = value.tag.type_id();
                if value.data.is_null() {
                    let _ = write!(output, "<{} type_id={} null>", kind, type_id);
                } else {
                    let _ = write!(output, "<{} type_id={} @{:p}>", kind, type_id, value.data);
                }
            }
        }
    }
}

/// Print any DynamicBox value (inspects type tag to determine format)
///
/// Supports ALL known ZRTL runtime types including Array, Tuple, Optional, etc.
///
/// # Safety
/// The pointer must point to a valid DynamicBox
#[no_mangle]
pub unsafe extern "C" fn io_print_dynamic(value_ptr: *const zrtl::DynamicBox) {
    if value_ptr.is_null() {
        print!("null");
        return;
    }
    let mut output = String::new();
    format_dynamic_box(&*value_ptr, &mut output);
    print!("{}", output);
}

/// Print any DynamicBox value with newline
#[no_mangle]
pub unsafe extern "C" fn io_println_dynamic(value_ptr: *const zrtl::DynamicBox) {
    if value_ptr.is_null() {
        println!("null");
        return;
    }
    let mut output = String::new();
    format_dynamic_box(&*value_ptr, &mut output);
    println!("{}", output);
}

/// Print any DynamicBox value to stderr
#[no_mangle]
pub unsafe extern "C" fn io_eprint_dynamic(value_ptr: *const zrtl::DynamicBox) {
    if value_ptr.is_null() {
        eprint!("null");
        return;
    }
    let mut output = String::new();
    format_dynamic_box(&*value_ptr, &mut output);
    eprint!("{}", output);
}

/// Print any DynamicBox value to stderr with newline
#[no_mangle]
pub unsafe extern "C" fn io_eprintln_dynamic(value_ptr: *const zrtl::DynamicBox) {
    if value_ptr.is_null() {
        eprintln!("null");
        return;
    }
    let mut output = String::new();
    format_dynamic_box(&*value_ptr, &mut output);
    eprintln!("{}", output);
}

/// Format a DynamicBox value to a ZRTL string
///
/// Returns a new ZRTL string that must be freed with string_free.
#[no_mangle]
pub unsafe extern "C" fn io_format_dynamic(value_ptr: *const zrtl::DynamicBox) -> StringPtr {
    if value_ptr.is_null() {
        return string_new("null");
    }
    let mut output = String::new();
    format_dynamic_box(&*value_ptr, &mut output);
    string_new(&output)
}

// ============================================================================
// Array Print Functions (direct, without DynamicBox wrapper)
// ============================================================================

/// Print a ZRTL array of f32 values
#[no_mangle]
pub unsafe extern "C" fn io_print_array_f32(arr: zrtl::ArrayConstPtr) {
    if arr.is_null() {
        print!("[]");
        return;
    }

    let len = zrtl::array_length(arr) as usize;
    let data: *const f32 = zrtl::array_data(arr);

    print!("[");
    for i in 0..len.min(10) {
        if i > 0 { print!(", "); }
        print!("{}", *data.add(i));
    }
    if len > 10 { print!(", ... ({} more)", len - 10); }
    print!("]");
}

/// Print a ZRTL array of i32 values
#[no_mangle]
pub unsafe extern "C" fn io_print_array_i32(arr: zrtl::ArrayConstPtr) {
    if arr.is_null() {
        print!("[]");
        return;
    }

    let len = zrtl::array_length(arr) as usize;
    let data: *const i32 = zrtl::array_data(arr);

    print!("[");
    for i in 0..len.min(10) {
        if i > 0 { print!(", "); }
        print!("{}", *data.add(i));
    }
    if len > 10 { print!(", ... ({} more)", len - 10); }
    print!("]");
}

/// Print a ZRTL array of i64 values
#[no_mangle]
pub unsafe extern "C" fn io_print_array_i64(arr: zrtl::ArrayConstPtr) {
    if arr.is_null() {
        print!("[]");
        return;
    }

    let len = zrtl::array_length(arr) as usize;
    let data: *const i64 = zrtl::array_data(arr);

    print!("[");
    for i in 0..len.min(10) {
        if i > 0 { print!(", "); }
        print!("{}", *data.add(i));
    }
    if len > 10 { print!(", ... ({} more)", len - 10); }
    print!("]");
}

/// Print a ZRTL array of f64 values
#[no_mangle]
pub unsafe extern "C" fn io_print_array_f64(arr: zrtl::ArrayConstPtr) {
    if arr.is_null() {
        print!("[]");
        return;
    }

    let len = zrtl::array_length(arr) as usize;
    let data: *const f64 = zrtl::array_data(arr);

    print!("[");
    for i in 0..len.min(10) {
        if i > 0 { print!(", "); }
        print!("{}", *data.add(i));
    }
    if len > 10 { print!(", ... ({} more)", len - 10); }
    print!("]");
}

/// Print a ZRTL array with newline (f32)
#[no_mangle]
pub unsafe extern "C" fn io_println_array_f32(arr: zrtl::ArrayConstPtr) {
    io_print_array_f32(arr);
    println!();
}

/// Print a ZRTL array with newline (i32)
#[no_mangle]
pub unsafe extern "C" fn io_println_array_i32(arr: zrtl::ArrayConstPtr) {
    io_print_array_i32(arr);
    println!();
}

/// Print a ZRTL array with newline (i64)
#[no_mangle]
pub unsafe extern "C" fn io_println_array_i64(arr: zrtl::ArrayConstPtr) {
    io_print_array_i64(arr);
    println!();
}

/// Print a ZRTL array with newline (f64)
#[no_mangle]
pub unsafe extern "C" fn io_println_array_f64(arr: zrtl::ArrayConstPtr) {
    io_print_array_f64(arr);
    println!();
}

// ============================================================================
// String Concatenation
// ============================================================================

/// Convert a string pointer to a DynamicBox (borrows the string, caller manages lifetime)
///
/// # Safety
/// - `s` must be a valid ZRTL string pointer or null
/// - The string must remain valid for the lifetime of the returned DynamicBox
/// - The caller must free the returned DynamicBox with `Box::from_raw`
#[no_mangle]
pub unsafe extern "C" fn io_string_to_dynamic(s: StringConstPtr) -> *mut zrtl::DynamicBox {
    use zrtl::{DynamicBox, TypeTag};

    if s.is_null() {
        return Box::into_raw(Box::new(DynamicBox::null()));
    }

    // Copy the string so the DynamicBox owns its data and can free it via dropper
    let copy = zrtl::string_copy(s);

    Box::into_raw(Box::new(DynamicBox {
        tag: TypeTag::STRING,
        size: std::mem::size_of::<*const u8>() as u32,
        data: copy as *mut u8,
        dropper: Some(drop_zrtl_string),
        display_fn: None,
    }))
}

/// Concatenate two ZRTL strings
///
/// Returns a new ZRTL string, caller must free with `string_free`.
#[no_mangle]
pub unsafe extern "C" fn io_string_concat(a: StringConstPtr, b: StringConstPtr) -> StringPtr {
    let mut result = String::new();

    if !a.is_null() {
        let len_a = string_length(a) as usize;
        let data_a = string_data(a);
        if len_a > 0 && !data_a.is_null() {
            let bytes = std::slice::from_raw_parts(data_a, len_a);
            if let Ok(s) = std::str::from_utf8(bytes) {
                result.push_str(s);
            }
        }
    }

    if !b.is_null() {
        let len_b = string_length(b) as usize;
        let data_b = string_data(b);
        if len_b > 0 && !data_b.is_null() {
            let bytes = std::slice::from_raw_parts(data_b, len_b);
            if let Ok(s) = std::str::from_utf8(bytes) {
                result.push_str(s);
            }
        }
    }

    string_new(&result)
}

/// Concatenate two values (DynamicBox) and return a DynamicBox containing the resulting string
/// Used by f-string desugaring so the result works with println_dynamic
#[no_mangle]
pub unsafe extern "C" fn io_concat_dynamic(a: *const zrtl::DynamicBox, b: *const zrtl::DynamicBox) -> *mut zrtl::DynamicBox {
    use zrtl::{DynamicBox, TypeTag, TypeCategory};

    let mut result = String::new();

    // Format first argument
    if !a.is_null() {
        let box_a = &*a;
        // Check if it's a string type - if so, extract directly
        if box_a.tag.category() == TypeCategory::String {
            let ptr = box_a.data as StringConstPtr;
            if !ptr.is_null() {
                let len = string_length(ptr) as usize;
                let data = string_data(ptr);
                if len > 0 && !data.is_null() {
                    let bytes = std::slice::from_raw_parts(data, len);
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        result.push_str(s);
                    }
                }
            }
        } else {
            // Use format_dynamic_box for other types
            format_dynamic_box(box_a, &mut result);
        }
    }

    // Format second argument
    if !b.is_null() {
        let box_b = &*b;
        if box_b.tag.category() == TypeCategory::String {
            let ptr = box_b.data as StringConstPtr;
            if !ptr.is_null() {
                let len = string_length(ptr) as usize;
                let data = string_data(ptr);
                if len > 0 && !data.is_null() {
                    let bytes = std::slice::from_raw_parts(data, len);
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        result.push_str(s);
                    }
                }
            }
        } else {
            format_dynamic_box(box_b, &mut result);
        }
    }

    // Create a new ZRTL string and wrap in DynamicBox
    let str_ptr = string_new(&result);

    Box::into_raw(Box::new(DynamicBox {
        tag: TypeTag::STRING,
        size: std::mem::size_of::<*const u8>() as u32,
        data: str_ptr as *mut u8,
        dropper: Some(drop_zrtl_string),
        display_fn: None,
    }))
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_io",
    symbols: [
        // String output (ZRTL string format) - takes StringConstPtr (i64 pointer)
        ("$IO$print", io_print, (i64) -> void),
        ("$IO$println", io_println, (i64) -> void),
        ("$IO$eprint", io_eprint, (i64) -> void),
        ("$IO$eprintln", io_eprintln, (i64) -> void),

        // Primitive output
        ("$IO$print_i64", io_print_i64),
        ("$IO$println_i64", io_println_i64),
        ("$IO$print_u64", io_print_u64),
        ("$IO$println_u64", io_println_u64),
        ("$IO$print_f64", io_print_f64),
        ("$IO$println_f64", io_println_f64),
        ("$IO$print_bool", io_print_bool),
        ("$IO$println_bool", io_println_bool),
        ("$IO$print_char", io_print_char),
        ("$IO$println_char", io_println_char),

        // Flush
        ("$IO$flush", io_flush),
        ("$IO$flush_stderr", io_flush_stderr),

        // Input (returns ZRTL strings)
        ("$IO$read_line", io_read_line),
        ("$IO$input", io_input),

        // Formatting (returns ZRTL strings)
        ("$IO$format_i64", io_format_i64),
        ("$IO$format_u64", io_format_u64),
        ("$IO$format_f64", io_format_f64),
        ("$IO$format_f64_precision", io_format_f64_precision),
        ("$IO$format_bool", io_format_bool),
        ("$IO$format_hex", io_format_hex),
        ("$IO$format_hex_upper", io_format_hex_upper),
        ("$IO$format_binary", io_format_binary),
        ("$IO$format_octal", io_format_octal),
        ("$IO$format_char", io_format_char),

        // Memory management
        ("$IO$string_free", io_string_free),

        // Dynamic (DynamicBox) printing - handles ALL ZRTL types
        // These functions expect DynamicBox, so compiler will auto-box arguments
        ("$IO$print_dynamic", io_print_dynamic, dynamic(1) -> void),
        ("$IO$println_dynamic", io_println_dynamic, dynamic(1) -> void),
        ("$IO$eprint_dynamic", io_eprint_dynamic, dynamic(1) -> void),
        ("$IO$eprintln_dynamic", io_eprintln_dynamic, dynamic(1) -> void),
        ("$IO$format_dynamic", io_format_dynamic, dynamic(1) -> dynamic),

        // Array printing (direct, type-specific)
        ("$IO$print_array_f32", io_print_array_f32),
        ("$IO$println_array_f32", io_println_array_f32),
        ("$IO$print_array_i32", io_print_array_i32),
        ("$IO$println_array_i32", io_println_array_i32),
        ("$IO$print_array_i64", io_print_array_i64),
        ("$IO$println_array_i64", io_println_array_i64),
        ("$IO$print_array_f64", io_print_array_f64),
        ("$IO$println_array_f64", io_println_array_f64),

        // String operations
        ("$IO$string_concat", io_string_concat, (i64, i64) -> i64),
        ("$IO$string_to_dynamic", io_string_to_dynamic, (i64) -> i64),
        ("$IO$concat_dynamic", io_concat_dynamic, dynamic(2) -> dynamic),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zrtl::{TypeCategory, TypeFlags, TypeTag, string_as_str};

    #[test]
    fn test_format_i64() {
        let result = io_format_i64(42);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(string_as_str(result), Some("42"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_f64() {
        let result = io_format_f64(12.34);
        assert!(!result.is_null());
        unsafe {
            let s = string_as_str(result);
            assert!(s.is_some());
            assert!(s.unwrap().starts_with("12.34"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_f64_precision() {
        let result = io_format_f64_precision(12.34567, 2);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(string_as_str(result), Some("12.35"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_hex() {
        let result = io_format_hex(255);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(string_as_str(result), Some("ff"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_binary() {
        let result = io_format_binary(5);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(string_as_str(result), Some("101"));
            string_free(result);
        }
    }

    #[test]
    fn test_format_bool() {
        let result_true = io_format_bool(true);
        let result_false = io_format_bool(false);
        assert!(!result_true.is_null());
        assert!(!result_false.is_null());
        unsafe {
            assert_eq!(string_as_str(result_true), Some("true"));
            assert_eq!(string_as_str(result_false), Some("false"));
            string_free(result_true);
            string_free(result_false);
        }
    }

    #[test]
    fn test_print_zrtl_string() {
        let s = string_new("Hello, ZRTL!");
        assert!(!s.is_null());
        // Just verify it doesn't crash
        unsafe {
            io_print(s);
            io_println(s);
            string_free(s);
        }
    }

    #[test]
    fn test_format_dynamic_opaque_includes_type_id_without_dereference() {
        // Deliberately use an invalid raw pointer value to verify formatting
        // does not dereference opaque/custom payload bytes.
        let boxed = zrtl::DynamicBox {
            tag: TypeTag::new(TypeCategory::Opaque, 0x1234, TypeFlags::NONE),
            size: 8,
            data: std::ptr::dangling_mut::<u8>(),
            dropper: None,
            display_fn: None,
        };

        let formatted = unsafe { io_format_dynamic(&boxed as *const zrtl::DynamicBox) };
        assert!(!formatted.is_null());
        unsafe {
            let rendered = string_as_str(formatted).unwrap_or_default();
            assert!(
                rendered.contains("type_id=4660"),
                "expected opaque type_id in output, got: {}",
                rendered
            );
            string_free(formatted);
        }
    }

    #[test]
    fn test_string_to_dynamic_copies_string() {
        // Verify io_string_to_dynamic creates an owned copy
        let original = string_new("hello");
        unsafe {
            let boxed_ptr = io_string_to_dynamic(original);
            assert!(!boxed_ptr.is_null());

            let boxed = &*boxed_ptr;
            assert_eq!(boxed.tag.category(), TypeCategory::String);
            assert!(boxed.dropper.is_some(), "DynamicBox should have a dropper");

            // The data pointer should be different from the original (it's a copy)
            assert_ne!(boxed.data as *const u8, original as *const u8 as *const u8);

            // Verify the copied string content
            let str_ptr = boxed.data as StringConstPtr;
            assert_eq!(string_as_str(str_ptr), Some("hello"));

            // Clean up: free the DynamicBox's string via dropper, the Box, and the original
            if let Some(dropper) = boxed.dropper {
                dropper(boxed.data);
            }
            let _ = Box::from_raw(boxed_ptr);
            string_free(original);
        }
    }

    #[test]
    fn test_string_to_dynamic_null_input() {
        unsafe {
            let boxed_ptr = io_string_to_dynamic(std::ptr::null());
            assert!(!boxed_ptr.is_null());
            let boxed = &*boxed_ptr;
            assert!(boxed.data.is_null(), "null input should produce null DynamicBox");
            let _ = Box::from_raw(boxed_ptr);
        }
    }

    #[test]
    fn test_concat_dynamic_has_dropper() {
        // Verify io_concat_dynamic creates DynamicBox with string dropper
        let s1 = string_new("hello ");
        let s2 = string_new("world");

        unsafe {
            let box1 = io_string_to_dynamic(s1);
            let box2 = io_string_to_dynamic(s2);

            let result = io_concat_dynamic(box1, box2);
            assert!(!result.is_null());

            let result_box = &*result;
            assert_eq!(result_box.tag.category(), TypeCategory::String);
            assert!(result_box.dropper.is_some(), "concat result should have a string dropper");

            // Verify content
            let str_ptr = result_box.data as StringConstPtr;
            assert_eq!(string_as_str(str_ptr), Some("hello world"));

            // Clean up (call dropper for the string, then free the Box)
            if let Some(dropper) = result_box.dropper {
                dropper(result_box.data);
            }
            let _ = Box::from_raw(result);

            // Clean up inputs
            let b1 = &*box1;
            if let Some(dropper) = b1.dropper { dropper(b1.data); }
            let _ = Box::from_raw(box1);
            let b2 = &*box2;
            if let Some(dropper) = b2.dropper { dropper(b2.data); }
            let _ = Box::from_raw(box2);

            string_free(s1);
            string_free(s2);
        }
    }

    #[test]
    fn test_format_dynamic_null_safety() {
        unsafe {
            // Null pointer should return "null"
            let formatted = io_format_dynamic(std::ptr::null());
            assert!(!formatted.is_null());
            assert_eq!(string_as_str(formatted), Some("null"));
            string_free(formatted);
        }
    }

    #[test]
    fn test_format_dynamic_null_data_box() {
        // DynamicBox with null data field
        let boxed = zrtl::DynamicBox::null();
        unsafe {
            let formatted = io_format_dynamic(&boxed as *const zrtl::DynamicBox);
            assert!(!formatted.is_null());
            assert_eq!(string_as_str(formatted), Some("null"));
            string_free(formatted);
        }
    }

    #[test]
    fn test_format_dynamic_int_types() {
        // Test various integer sizes
        let val: i32 = -42;
        let boxed = zrtl::DynamicBox {
            tag: TypeTag::new(TypeCategory::Int, 0, TypeFlags::NONE),
            size: 4,
            data: &val as *const i32 as *mut u8,
            dropper: None,
            display_fn: None,
        };
        unsafe {
            let formatted = io_format_dynamic(&boxed as *const zrtl::DynamicBox);
            assert_eq!(string_as_str(formatted), Some("-42"));
            string_free(formatted);
        }

        let val64: i64 = 9999999;
        let boxed64 = zrtl::DynamicBox {
            tag: TypeTag::new(TypeCategory::Int, 0, TypeFlags::NONE),
            size: 8,
            data: &val64 as *const i64 as *mut u8,
            dropper: None,
            display_fn: None,
        };
        unsafe {
            let formatted = io_format_dynamic(&boxed64 as *const zrtl::DynamicBox);
            assert_eq!(string_as_str(formatted), Some("9999999"));
            string_free(formatted);
        }
    }

    #[test]
    fn test_format_dynamic_float_types() {
        let val: f64 = 3.14;
        let boxed = zrtl::DynamicBox {
            tag: TypeTag::new(TypeCategory::Float, 0, TypeFlags::NONE),
            size: 8,
            data: &val as *const f64 as *mut u8,
            dropper: None,
            display_fn: None,
        };
        unsafe {
            let formatted = io_format_dynamic(&boxed as *const zrtl::DynamicBox);
            let s = string_as_str(formatted).unwrap_or_default();
            assert!(s.starts_with("3.14"), "expected 3.14..., got: {}", s);
            string_free(formatted);
        }
    }

    #[test]
    fn test_format_dynamic_bool() {
        let val: u8 = 1;
        let boxed = zrtl::DynamicBox {
            tag: TypeTag::new(TypeCategory::Bool, 0, TypeFlags::NONE),
            size: 1,
            data: &val as *const u8 as *mut u8,
            dropper: None,
            display_fn: None,
        };
        unsafe {
            let formatted = io_format_dynamic(&boxed as *const zrtl::DynamicBox);
            assert_eq!(string_as_str(formatted), Some("true"));
            string_free(formatted);
        }
    }

    #[test]
    fn test_format_dynamic_string() {
        let s = string_new("test string");
        let boxed = zrtl::DynamicBox {
            tag: TypeTag::STRING,
            size: std::mem::size_of::<*const u8>() as u32,
            data: s as *mut u8,
            dropper: None,
            display_fn: None,
        };
        unsafe {
            let formatted = io_format_dynamic(&boxed as *const zrtl::DynamicBox);
            assert_eq!(string_as_str(formatted), Some("test string"));
            string_free(formatted);
            string_free(s);
        }
    }
}
