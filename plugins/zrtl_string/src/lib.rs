//! ZRTL String Plugin
//!
//! Provides string manipulation functions for Zyntax-based languages.
//!
//! ## Exported Symbols
//!
//! ### Basic Operations
//! - `$String$length` - Get string length
//! - `$String$is_empty` - Check if string is empty
//! - `$String$concat` - Concatenate two strings
//! - `$String$repeat` - Repeat string n times
//!
//! ### Case Conversion
//! - `$String$to_upper`, `$String$to_lower` - Case conversion
//! - `$String$capitalize` - Capitalize first letter
//!
//! ### Trimming
//! - `$String$trim`, `$String$trim_start`, `$String$trim_end` - Whitespace trimming
//!
//! ### Search & Replace
//! - `$String$contains`, `$String$starts_with`, `$String$ends_with` - Search
//! - `$String$index_of`, `$String$last_index_of` - Find position
//! - `$String$replace`, `$String$replace_all` - Replace substrings
//!
//! ### Extraction
//! - `$String$substring`, `$String$char_at` - Extract parts
//! - `$String$split` - Split into array
//!
//! ### Conversion
//! - `$String$parse_int`, `$String$parse_float` - Parse numbers
//! - `$String$from_int`, `$String$from_float` - Convert to string

use zrtl::{
    array_new, array_push, string_as_str, string_data, string_length, string_new, zrtl_plugin,
    ArrayPtr, StringPtr,
};

// ============================================================================
// Basic Operations
// ============================================================================

/// Get string length in bytes
#[no_mangle]
pub extern "C" fn string_len(s: StringPtr) -> i64 {
    unsafe { string_length(s) as i64 }
}

/// Get string length in characters (Unicode-aware)
#[no_mangle]
pub extern "C" fn string_char_count(s: StringPtr) -> i64 {
    match unsafe { string_as_str(s) } {
        Some(s) => s.chars().count() as i64,
        None => 0,
    }
}

/// Check if string is empty
#[no_mangle]
pub extern "C" fn string_is_empty(s: StringPtr) -> i32 {
    unsafe { (string_length(s) == 0) as i32 }
}

/// Concatenate two strings
#[no_mangle]
pub extern "C" fn string_concat(a: StringPtr, b: StringPtr) -> StringPtr {
    let a_str = unsafe { string_as_str(a) }.unwrap_or("");
    let b_str = unsafe { string_as_str(b) }.unwrap_or("");
    string_new(&format!("{}{}", a_str, b_str))
}

/// Repeat string n times
#[no_mangle]
pub extern "C" fn string_repeat(s: StringPtr, n: i64) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    if n <= 0 {
        return string_new("");
    }
    string_new(&s_str.repeat(n as usize))
}

/// Join array of strings with separator
#[no_mangle]
pub extern "C" fn string_join(arr: ArrayPtr, sep: StringPtr) -> StringPtr {
    let sep_str = unsafe { string_as_str(sep) }.unwrap_or("");

    // Read array header
    if arr.is_null() {
        return string_new("");
    }

    let header = arr as *const i32;
    let _capacity = unsafe { *header };
    let length = unsafe { *header.add(1) };
    let data = unsafe { header.add(2) } as *const StringPtr;

    let mut parts = Vec::new();
    for i in 0..length as usize {
        let ptr = unsafe { *data.add(i) };
        if let Some(s) = unsafe { string_as_str(ptr) } {
            parts.push(s.to_string());
        }
    }

    string_new(&parts.join(sep_str))
}

/// Join `n` strings held in a contiguous array of string pointers, the
/// layout of a compiled `List<String>`'s data, with a separator. One
/// allocation for the whole result.
#[no_mangle]
pub extern "C" fn string_join_n(data: *const StringPtr, n: i64, sep: StringPtr) -> StringPtr {
    let sep_str = unsafe { string_as_str(sep) }.unwrap_or("");
    if data.is_null() || n <= 0 {
        return string_new("");
    }
    let mut total = 0usize;
    for i in 0..n as usize {
        let ptr = unsafe { *data.add(i) };
        total += unsafe { string_length(ptr) } as usize;
    }
    total += sep_str.len() * (n as usize - 1);
    let mut out = String::with_capacity(total);
    for i in 0..n as usize {
        if i > 0 {
            out.push_str(sep_str);
        }
        let ptr = unsafe { *data.add(i) };
        if let Some(part) = unsafe { string_as_str(ptr) } {
            out.push_str(part);
        }
    }
    string_new(&out)
}

// ============================================================================
// Case Conversion
// ============================================================================

/// Convert to uppercase
#[no_mangle]
pub extern "C" fn string_to_upper(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(&s.to_uppercase()),
        None => string_new(""),
    }
}

/// Convert to lowercase
#[no_mangle]
pub extern "C" fn string_to_lower(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(&s.to_lowercase()),
        None => string_new(""),
    }
}

/// Capitalize first character
#[no_mangle]
pub extern "C" fn string_capitalize(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) if !s.is_empty() => {
            let mut chars = s.chars();
            let first = chars.next().unwrap().to_uppercase();
            let rest: String = chars.collect();
            string_new(&format!("{}{}", first, rest))
        }
        _ => string_new(""),
    }
}

/// Convert to title case (capitalize each word)
#[no_mangle]
pub extern "C" fn string_to_title(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => {
            let result: String = s
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        Some(c) => format!("{}{}", c.to_uppercase(), chars.as_str().to_lowercase()),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            string_new(&result)
        }
        None => string_new(""),
    }
}

// ============================================================================
// Trimming
// ============================================================================

/// Trim whitespace from both ends
#[no_mangle]
pub extern "C" fn string_trim(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(s.trim()),
        None => string_new(""),
    }
}

/// Trim whitespace from start
#[no_mangle]
pub extern "C" fn string_trim_start(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(s.trim_start()),
        None => string_new(""),
    }
}

/// Trim whitespace from end
#[no_mangle]
pub extern "C" fn string_trim_end(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(s.trim_end()),
        None => string_new(""),
    }
}

// ============================================================================
// Search
// ============================================================================

/// Check if string contains substring
#[no_mangle]
pub extern "C" fn string_contains(haystack: StringPtr, needle: StringPtr) -> i32 {
    let h = unsafe { string_as_str(haystack) }.unwrap_or("");
    let n = unsafe { string_as_str(needle) }.unwrap_or("");
    h.contains(n) as i32
}

/// Check if string starts with prefix
#[no_mangle]
pub extern "C" fn string_starts_with(s: StringPtr, prefix: StringPtr) -> i32 {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let p_str = unsafe { string_as_str(prefix) }.unwrap_or("");
    s_str.starts_with(p_str) as i32
}

/// Check if string ends with suffix
#[no_mangle]
pub extern "C" fn string_ends_with(s: StringPtr, suffix: StringPtr) -> i32 {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let suf_str = unsafe { string_as_str(suffix) }.unwrap_or("");
    s_str.ends_with(suf_str) as i32
}

/// Find first index of substring, returns -1 if not found
#[no_mangle]
pub extern "C" fn string_index_of(haystack: StringPtr, needle: StringPtr) -> i64 {
    let h = unsafe { string_as_str(haystack) }.unwrap_or("");
    let n = unsafe { string_as_str(needle) }.unwrap_or("");
    match h.find(n) {
        Some(i) => i as i64,
        None => -1,
    }
}

/// Find the first byte index of `needle` at or after byte `from`, or
/// -1. `from` is clamped to the haystack and rounded up to a character
/// boundary.
#[no_mangle]
pub extern "C" fn string_index_of_from(haystack: StringPtr, needle: StringPtr, from: i64) -> i64 {
    // Over the bytes: a search that resumes along a long text must not
    // re-read the whole of it each time, which decoding it would.
    let h_len = unsafe { string_length(haystack) }.max(0) as usize;
    if haystack.is_null() {
        return -1;
    }
    let h = unsafe { std::slice::from_raw_parts(string_data(haystack), h_len) };
    let n = unsafe { string_as_str(needle) }.unwrap_or("").as_bytes();
    let mut from = (from.max(0) as usize).min(h.len());
    // A continuation byte is never a match start.
    while from < h.len() && (h[from] & 0xC0) == 0x80 {
        from += 1;
    }
    if n.is_empty() {
        return from as i64;
    }
    if from + n.len() > h.len() {
        return -1;
    }
    let rest = &h[from..];
    let found = if n.len() == 1 {
        rest.iter().position(|&b| b == n[0])
    } else {
        rest.windows(n.len()).position(|w| w == n)
    };
    match found {
        Some(i) => (from + i) as i64,
        None => -1,
    }
}

/// Find last index of substring, returns -1 if not found
#[no_mangle]
pub extern "C" fn string_last_index_of(haystack: StringPtr, needle: StringPtr) -> i64 {
    let h = unsafe { string_as_str(haystack) }.unwrap_or("");
    let n = unsafe { string_as_str(needle) }.unwrap_or("");
    match h.rfind(n) {
        Some(i) => i as i64,
        None => -1,
    }
}

/// Count occurrences of substring
#[no_mangle]
pub extern "C" fn string_count(haystack: StringPtr, needle: StringPtr) -> i64 {
    let h = unsafe { string_as_str(haystack) }.unwrap_or("");
    let n = unsafe { string_as_str(needle) }.unwrap_or("");
    if n.is_empty() {
        return 0;
    }
    h.matches(n).count() as i64
}

// ============================================================================
// Replace
// ============================================================================

/// Replace first occurrence of pattern with replacement
#[no_mangle]
pub extern "C" fn string_replace(s: StringPtr, from: StringPtr, to: StringPtr) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let from_str = unsafe { string_as_str(from) }.unwrap_or("");
    let to_str = unsafe { string_as_str(to) }.unwrap_or("");
    string_new(&s_str.replacen(from_str, to_str, 1))
}

/// Replace all occurrences of pattern with replacement
#[no_mangle]
pub extern "C" fn string_replace_all(s: StringPtr, from: StringPtr, to: StringPtr) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let from_str = unsafe { string_as_str(from) }.unwrap_or("");
    let to_str = unsafe { string_as_str(to) }.unwrap_or("");
    string_new(&s_str.replace(from_str, to_str))
}

/// Remove all occurrences of substring
#[no_mangle]
pub extern "C" fn string_remove(s: StringPtr, pattern: StringPtr) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let p_str = unsafe { string_as_str(pattern) }.unwrap_or("");
    string_new(&s_str.replace(p_str, ""))
}

// ============================================================================
// Extraction
// ============================================================================

/// Get substring from start to end (exclusive)
#[no_mangle]
pub extern "C" fn string_substring(s: StringPtr, start: i64, end: i64) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let len = s_str.len() as i64;

    let start = start.max(0).min(len) as usize;
    let end = end.max(0).min(len) as usize;

    if start >= end {
        return string_new("");
    }

    // Handle UTF-8 properly by using char indices
    let chars: Vec<char> = s_str.chars().collect();
    let start = start.min(chars.len());
    let end = end.min(chars.len());

    string_new(&chars[start..end].iter().collect::<String>())
}

/// Get character at index (returns empty string if out of bounds)
#[no_mangle]
pub extern "C" fn string_char_at(s: StringPtr, index: i64) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    match s_str.chars().nth(index as usize) {
        Some(c) => string_new(&c.to_string()),
        None => string_new(""),
    }
}

/// The character whose first byte is at byte offset `pos`, as a string of
/// its own. An offset at or past the end, or one inside a character,
/// gives an empty string. Constant time, unlike indexing by character.
#[no_mangle]
pub extern "C" fn string_char_at_byte(s: StringPtr, pos: i64) -> StringPtr {
    let len = unsafe { string_length(s) } as i64;
    if s.is_null() || pos < 0 || pos >= len {
        return string_new("");
    }
    let data = unsafe { string_data(s) };
    let rest = unsafe { std::slice::from_raw_parts(data.add(pos as usize), (len - pos) as usize) };
    let width = utf8_width(rest[0]).min(rest.len());
    match std::str::from_utf8(&rest[..width]) {
        Ok(c) => string_new(c),
        Err(_) => string_new(""),
    }
}

/// The bytes from offset `start` to `end`, as a string. Both must sit on
/// character boundaries, as the offsets `string_next_byte` hands out do.
#[no_mangle]
pub extern "C" fn string_bytes(s: StringPtr, start: i64, end: i64) -> StringPtr {
    let len = unsafe { string_length(s) } as i64;
    let start = start.clamp(0, len);
    let end = end.clamp(start, len);
    if s.is_null() || start == end {
        return string_new("");
    }
    let data = unsafe { string_data(s) };
    let bytes =
        unsafe { std::slice::from_raw_parts(data.add(start as usize), (end - start) as usize) };
    match std::str::from_utf8(bytes) {
        Ok(text) => string_new(text),
        Err(_) => string_new(""),
    }
}

/// The byte offset of the character after the one at `pos`; the string's
/// byte length once there is none.
#[no_mangle]
pub extern "C" fn string_next_byte(s: StringPtr, pos: i64) -> i64 {
    let len = unsafe { string_length(s) } as i64;
    if s.is_null() || pos < 0 || pos >= len {
        return len;
    }
    let lead = unsafe { *string_data(s).add(pos as usize) };
    (pos + utf8_width(lead) as i64).min(len)
}

/// How many bytes a UTF-8 character occupies, from its first byte.
fn utf8_width(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

/// Get character code at index (returns -1 if out of bounds)
#[no_mangle]
pub extern "C" fn string_char_code_at(s: StringPtr, index: i64) -> i32 {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    match s_str.chars().nth(index as usize) {
        Some(c) => c as i32,
        None => -1,
    }
}

/// Create string from character code
#[no_mangle]
pub extern "C" fn string_from_char_code(code: i32) -> StringPtr {
    match char::from_u32(code as u32) {
        Some(c) => string_new(&c.to_string()),
        None => string_new(""),
    }
}

/// Split string by delimiter into array
#[no_mangle]
pub extern "C" fn string_split(s: StringPtr, delimiter: StringPtr) -> ArrayPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let d_str = unsafe { string_as_str(delimiter) }.unwrap_or("");

    let parts: Vec<&str> = if d_str.is_empty() {
        // Split into characters
        s_str.split("").filter(|s| !s.is_empty()).collect()
    } else {
        s_str.split(d_str).collect()
    };

    let arr = array_new::<StringPtr>(parts.len());
    for part in parts {
        let part_ptr = string_new(part);
        unsafe {
            array_push(arr, part_ptr);
        }
    }
    arr
}

/// Split string into lines
#[no_mangle]
pub extern "C" fn string_lines(s: StringPtr) -> ArrayPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let parts: Vec<&str> = s_str.lines().collect();

    let arr = array_new::<StringPtr>(parts.len());
    for part in parts {
        let part_ptr = string_new(part);
        unsafe {
            array_push(arr, part_ptr);
        }
    }
    arr
}

// ============================================================================
// Padding
// ============================================================================

/// Pad string on the left to reach target length
#[no_mangle]
pub extern "C" fn string_pad_start(
    s: StringPtr,
    target_len: i64,
    pad_char: StringPtr,
) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let pad = unsafe { string_as_str(pad_char) }
        .and_then(|p| p.chars().next())
        .unwrap_or(' ');

    let current_len = s_str.chars().count();
    let target = target_len as usize;

    if current_len >= target {
        return string_new(s_str);
    }

    let padding: String = std::iter::repeat(pad).take(target - current_len).collect();
    string_new(&format!("{}{}", padding, s_str))
}

/// Pad string on the right to reach target length
#[no_mangle]
pub extern "C" fn string_pad_end(s: StringPtr, target_len: i64, pad_char: StringPtr) -> StringPtr {
    let s_str = unsafe { string_as_str(s) }.unwrap_or("");
    let pad = unsafe { string_as_str(pad_char) }
        .and_then(|p| p.chars().next())
        .unwrap_or(' ');

    let current_len = s_str.chars().count();
    let target = target_len as usize;

    if current_len >= target {
        return string_new(s_str);
    }

    let padding: String = std::iter::repeat(pad).take(target - current_len).collect();
    string_new(&format!("{}{}", s_str, padding))
}

// ============================================================================
// Conversion
// ============================================================================

/// Parse string as integer (returns 0 on failure)
#[no_mangle]
pub extern "C" fn string_parse_int(s: StringPtr) -> i64 {
    unsafe { string_as_str(s) }
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0)
}

/// Parse string as integer with radix (returns 0 on failure)
#[no_mangle]
pub extern "C" fn string_parse_int_radix(s: StringPtr, radix: i32) -> i64 {
    unsafe { string_as_str(s) }
        .and_then(|s| i64::from_str_radix(s.trim(), radix as u32).ok())
        .unwrap_or(0)
}

/// Parse string as float (returns 0.0 on failure)
#[no_mangle]
pub extern "C" fn string_parse_float(s: StringPtr) -> f64 {
    unsafe { string_as_str(s) }
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Convert integer to string
#[no_mangle]
pub extern "C" fn string_from_int(n: i64) -> StringPtr {
    string_new(&n.to_string())
}

/// Convert integer to string with radix
#[no_mangle]
pub extern "C" fn string_from_int_radix(n: i64, radix: i32) -> StringPtr {
    match radix {
        2 => string_new(&format!("{:b}", n)),
        8 => string_new(&format!("{:o}", n)),
        16 => string_new(&format!("{:x}", n)),
        _ => string_new(&n.to_string()),
    }
}

/// Convert float to string
#[no_mangle]
pub extern "C" fn string_from_float(n: f64) -> StringPtr {
    string_new(&n.to_string())
}

/// Convert float to string with precision
#[no_mangle]
pub extern "C" fn string_from_float_precision(n: f64, precision: i32) -> StringPtr {
    string_new(&format!("{:.prec$}", n, prec = precision as usize))
}

// ============================================================================
// Comparison
// ============================================================================

/// Compare two strings (returns -1, 0, or 1)
#[no_mangle]
pub extern "C" fn string_compare(a: StringPtr, b: StringPtr) -> i32 {
    let a_str = unsafe { string_as_str(a) }.unwrap_or("");
    let b_str = unsafe { string_as_str(b) }.unwrap_or("");
    match a_str.cmp(b_str) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Compare two strings ignoring case
#[no_mangle]
pub extern "C" fn string_compare_ignore_case(a: StringPtr, b: StringPtr) -> i32 {
    let a_str = unsafe { string_as_str(a) }.unwrap_or("").to_lowercase();
    let b_str = unsafe { string_as_str(b) }.unwrap_or("").to_lowercase();
    match a_str.cmp(&b_str) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Check if two strings are equal
#[no_mangle]
pub extern "C" fn string_equals(a: StringPtr, b: StringPtr) -> i32 {
    let a_str = unsafe { string_as_str(a) }.unwrap_or("");
    let b_str = unsafe { string_as_str(b) }.unwrap_or("");
    (a_str == b_str) as i32
}

/// Check if two strings are equal ignoring case
#[no_mangle]
pub extern "C" fn string_equals_ignore_case(a: StringPtr, b: StringPtr) -> i32 {
    let a_str = unsafe { string_as_str(a) }.unwrap_or("").to_lowercase();
    let b_str = unsafe { string_as_str(b) }.unwrap_or("").to_lowercase();
    (a_str == b_str) as i32
}

// ============================================================================
// Reverse
// ============================================================================

/// Reverse a string
#[no_mangle]
pub extern "C" fn string_reverse(s: StringPtr) -> StringPtr {
    match unsafe { string_as_str(s) } {
        Some(s) => string_new(&s.chars().rev().collect::<String>()),
        None => string_new(""),
    }
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_string",
    symbols: [
        // Basic
        ("$String$length", string_len),
        ("$String$char_count", string_char_count),
        ("$String$is_empty", string_is_empty),
        ("$String$concat", string_concat),
        ("$String$repeat", string_repeat),
        ("$String$join", string_join),
        ("$String$join_n", string_join_n),

        // Case
        ("$String$to_upper", string_to_upper),
        ("$String$to_lower", string_to_lower),
        ("$String$capitalize", string_capitalize),
        ("$String$to_title", string_to_title),

        // Trim
        ("$String$trim", string_trim),
        ("$String$trim_start", string_trim_start),
        ("$String$trim_end", string_trim_end),

        // Search
        ("$String$contains", string_contains),
        ("$String$starts_with", string_starts_with),
        ("$String$ends_with", string_ends_with),
        ("$String$index_of", string_index_of),
        ("$String$index_of_from", string_index_of_from),
        ("$String$last_index_of", string_last_index_of),
        ("$String$count", string_count),

        // Replace
        ("$String$replace", string_replace),
        ("$String$replace_all", string_replace_all),
        ("$String$remove", string_remove),

        // Extract
        ("$String$substring", string_substring),
        ("$String$char_at", string_char_at),
        ("$String$char_at_byte", string_char_at_byte),
        ("$String$next_byte", string_next_byte),
        ("$String$bytes", string_bytes),
        ("$String$char_code_at", string_char_code_at),
        ("$String$from_char_code", string_from_char_code),
        ("$String$split", string_split),
        ("$String$lines", string_lines),

        // Padding
        ("$String$pad_start", string_pad_start),
        ("$String$pad_end", string_pad_end),

        // Conversion
        ("$String$parse_int", string_parse_int),
        ("$String$parse_int_radix", string_parse_int_radix),
        ("$String$parse_float", string_parse_float),
        ("$String$from_int", string_from_int),
        ("$String$from_int_radix", string_from_int_radix),
        ("$String$from_float", string_from_float),
        ("$String$from_float_precision", string_from_float_precision),

        // Comparison
        ("$String$compare", string_compare),
        ("$String$compare_ignore_case", string_compare_ignore_case),
        ("$String$equals", string_equals),
        ("$String$equals_ignore_case", string_equals_ignore_case),

        // Other
        ("$String$reverse", string_reverse),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let s = string_new("hello");
        assert_eq!(string_len(s), 5);
        assert_eq!(string_is_empty(s), 0);

        let empty = string_new("");
        assert_eq!(string_is_empty(empty), 1);
    }

    #[test]
    fn test_concat() {
        let a = string_new("hello");
        let b = string_new(" world");
        let result = string_concat(a, b);
        assert_eq!(unsafe { string_as_str(result) }, Some("hello world"));
    }

    #[test]
    fn test_case() {
        let s = string_new("Hello World");

        let upper = string_to_upper(s);
        assert_eq!(unsafe { string_as_str(upper) }, Some("HELLO WORLD"));

        let lower = string_to_lower(s);
        assert_eq!(unsafe { string_as_str(lower) }, Some("hello world"));
    }

    #[test]
    fn test_trim() {
        let s = string_new("  hello  ");

        let trimmed = string_trim(s);
        assert_eq!(unsafe { string_as_str(trimmed) }, Some("hello"));
    }

    #[test]
    fn test_search() {
        let s = string_new("hello world");
        let needle = string_new("world");

        assert_eq!(string_contains(s, needle), 1);
        assert_eq!(string_index_of(s, needle), 6);
    }

    #[test]
    fn test_replace() {
        let s = string_new("hello world");
        let from = string_new("world");
        let to = string_new("rust");

        let result = string_replace_all(s, from, to);
        assert_eq!(unsafe { string_as_str(result) }, Some("hello rust"));
    }

    #[test]
    fn test_substring() {
        let s = string_new("hello world");
        let sub = string_substring(s, 0, 5);
        assert_eq!(unsafe { string_as_str(sub) }, Some("hello"));
    }

    #[test]
    fn test_parse() {
        let s = string_new("42");
        assert_eq!(string_parse_int(s), 42);

        let f = string_new("3.14");
        assert!((string_parse_float(f) - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_from() {
        let s = string_from_int(42);
        assert_eq!(unsafe { string_as_str(s) }, Some("42"));

        let hex = string_from_int_radix(255, 16);
        assert_eq!(unsafe { string_as_str(hex) }, Some("ff"));
    }

    #[test]
    fn test_reverse() {
        let s = string_new("hello");
        let rev = string_reverse(s);
        assert_eq!(unsafe { string_as_str(rev) }, Some("olleh"));
    }

    #[test]
    fn test_padding() {
        let s = string_new("42");
        let pad = string_new("0");

        let padded = string_pad_start(s, 5, pad);
        assert_eq!(unsafe { string_as_str(padded) }, Some("00042"));
    }
}
