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

use zrtl::string::{string_as_bytes, string_build, string_flags, string_from_bytes, TEXT};
use zrtl::{
    array_new, array_push, string_as_str, string_data, string_length, string_new, zrtl_plugin,
    ArrayPtr, StringPtr,
};

/// Whether a string is TEXT; null is the empty text.
fn is_text(s: StringPtr) -> bool {
    s.is_null() || unsafe { string_flags(s) } & TEXT != 0
}

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
    // The header carries the count.
    unsafe { zrtl::string::string_char_count(s) as i64 }
}

/// Check if string is empty
#[no_mangle]
pub extern "C" fn string_is_empty(s: StringPtr) -> i32 {
    unsafe { (string_length(s) == 0) as i32 }
}

/// Concatenate two strings
#[no_mangle]
pub extern "C" fn string_concat(a: StringPtr, b: StringPtr) -> StringPtr {
    zrtl::string::string_concat(a, b)
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
    if data.is_null() || n <= 0 {
        return string_new("");
    }
    unsafe { join_parts(std::slice::from_raw_parts(data, n as usize), sep) }
}

/// `parts` joined with `sep`: TEXT when all of them are.
///
/// # Safety
/// Every part and the separator must be null or valid strings.
unsafe fn join_parts(parts: &[StringPtr], sep: StringPtr) -> StringPtr {
    let n = parts.len();
    let sep_bytes = string_as_bytes(sep);
    let sep_chars = zrtl::string::string_char_count(sep);
    let mut total = sep_bytes.len() * (n - 1);
    let mut chars = sep_chars * (n - 1);
    let mut text = is_text(sep);
    for &p in parts {
        total += string_length(p) as usize;
        chars += zrtl::string::string_char_count(p);
        text &= is_text(p);
    }
    let fill = |mut at: *mut u8| {
        for (i, &p) in parts.iter().enumerate() {
            if i > 0 {
                std::ptr::copy_nonoverlapping(sep_bytes.as_ptr(), at, sep_bytes.len());
                at = at.add(sep_bytes.len());
            }
            let part = string_as_bytes(p);
            std::ptr::copy_nonoverlapping(part.as_ptr(), at, part.len());
            at = at.add(part.len());
        }
    };
    if text {
        return string_build(total, chars, true, fill);
    }
    // A part that is not TEXT: the joined bytes decide.
    let mut joined = vec![0u8; total];
    fill(joined.as_mut_ptr());
    string_from_bytes(&joined)
}

// ============================================================================
// Case Conversion
// ============================================================================

/// Convert to uppercase: Unicode casing on TEXT, ASCII bytewise on a
/// byte string.
#[no_mangle]
pub extern "C" fn string_to_upper(s: StringPtr) -> StringPtr {
    let bytes = unsafe { string_as_bytes(s) };
    if !is_text(s) {
        return zrtl::bytes_new(&bytes.to_ascii_uppercase());
    }
    // SAFETY: a TEXT string holds UTF-8.
    string_new(&unsafe { std::str::from_utf8_unchecked(bytes) }.to_uppercase())
}

/// Convert to lowercase: Unicode casing on TEXT, ASCII bytewise on a
/// byte string.
#[no_mangle]
pub extern "C" fn string_to_lower(s: StringPtr) -> StringPtr {
    let bytes = unsafe { string_as_bytes(s) };
    if !is_text(s) {
        return zrtl::bytes_new(&bytes.to_ascii_lowercase());
    }
    // SAFETY: a TEXT string holds UTF-8.
    string_new(&unsafe { std::str::from_utf8_unchecked(bytes) }.to_lowercase())
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

/// A hash of the bytes, the same for equal contents; FNV-1a.
#[no_mangle]
pub extern "C" fn string_hash(s: StringPtr) -> i64 {
    if s.is_null() {
        return 0;
    }
    zrtl::fnv1a_bytes(unsafe { string_as_bytes(s) }) as i64
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
    unsafe { substring_of(s, start, end) }
}

/// Characters `start..end` of a string, clamped into it.
///
/// # Safety
/// `s` must be null or a valid string.
unsafe fn substring_of(s: StringPtr, start: i64, end: i64) -> StringPtr {
    if !s.is_null() && is_text(s) {
        // Character positions to byte offsets through the header.
        let count = zrtl::string::string_char_count(s) as i64;
        let (start, end) = (start.clamp(0, count), end.clamp(0, count));
        if start >= end {
            return string_new("");
        }
        let from = zrtl::string::string_char_offset(s, start as usize).unwrap_or(0);
        let to = zrtl::string::string_char_offset(s, end as usize).unwrap_or(from);
        return zrtl::string::string_slice(s, from, to);
    }
    let s_str = string_as_str(s).unwrap_or("");
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
    // The character's bytes, found through the string's index when it
    // is long; a copy of them as a string of its own.
    let Ok(index) = usize::try_from(index) else {
        return string_new("");
    };
    match unsafe { zrtl::string::string_char_range(s, index) } {
        Some(range) => unsafe { zrtl::string::string_slice(s, range.start, range.end) },
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
    if is_text(s) {
        if rest[0] & 0xC0 == 0x80 {
            return string_new("");
        }
        let pos = pos as usize;
        return unsafe { zrtl::string::string_slice(s, pos, pos + width) };
    }
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
    // A cut at a character boundary, as the callers make; a cut inside
    // a character is the empty string rather than broken text.
    let starts_inside = bytes.first().is_some_and(|b| b & 0xC0 == 0x80);
    let ends_inside =
        (end as usize) < len as usize && unsafe { *data.add(end as usize) } & 0xC0 == 0x80;
    if starts_inside || ends_inside {
        return string_new("");
    }
    unsafe { zrtl::string::string_slice(s, start as usize, end as usize) }
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
    unsafe { char_code_of(s, index) }
}

/// The code of character `index`, -1 out of range.
///
/// # Safety
/// `s` must be null or a valid string.
unsafe fn char_code_of(s: StringPtr, index: i64) -> i32 {
    if !s.is_null() && is_text(s) {
        let Ok(index) = usize::try_from(index) else {
            return -1;
        };
        return match zrtl::string::string_char_range(s, index) {
            Some(range) => string_as_str(s)
                .and_then(|t| t[range].chars().next())
                .map_or(-1, |c| c as i32),
            None => -1,
        };
    }
    let s_str = string_as_str(s).unwrap_or("");
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
    // Digits written into a stack buffer, then one allocation.
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let negative = n < 0;
    let mut m = n.unsigned_abs();
    loop {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
        if m == 0 {
            break;
        }
    }
    if negative {
        i -= 1;
        buf[i] = b'-';
    }
    string_from_bytes(&buf[i..])
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
    // Byte order is code point order for UTF-8, so no decoding.
    let a = unsafe { string_as_bytes(a) };
    let b = unsafe { string_as_bytes(b) };
    match a.cmp(b) {
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
    unsafe { zrtl::string::string_equals(a, b) as i32 }
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
        ("$String$hash", string_hash),
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
        let t = string_new(&"héllo wörld ".repeat(10));
        let sub = string_substring(t, 13, 18);
        assert_eq!(unsafe { string_as_str(sub) }, Some("éllo "));
        assert_eq!(
            unsafe { string_as_str(string_substring(t, 115, 200)) },
            Some("örld ")
        );
        assert_eq!(string_char_code_at(t, 13), 'é' as i32);
        assert_eq!(string_char_code_at(t, 120), -1);
        assert_eq!(unsafe { string_as_str(string_char_at(t, 1)) }, Some("é"));
        assert_eq!(string_char_at(t, 1), zrtl::string::char_text(0xE9));
    }

    #[test]
    fn test_header_flags_carry_through() {
        use zrtl::string::{string_header, INDEXED, TEXT};
        let a = string_new(&"é".repeat(40));
        let b = string_new(&"x".repeat(40));
        let joined = string_concat(a, b);
        let h = unsafe { string_header(joined) };
        assert_eq!((h.byte_len, h.char_len), (120, 80));
        assert_eq!(h.flags & (TEXT | INDEXED), TEXT | INDEXED);
        assert_eq!(string_char_count(joined), 80);
        let parts = [a, b, a];
        let sep = string_new("-");
        let j = string_join_n(parts.as_ptr(), 3, sep);
        assert_eq!(string_char_count(j), 122);
        assert_eq!(string_len(j), 202);
        let raw = zrtl::bytes_new(&[0xff, b'a']);
        let up = string_to_upper(raw);
        assert_eq!(unsafe { string_as_bytes(up) }, &[0xff, b'A']);
        assert_eq!(unsafe { string_header(up) }.flags & TEXT, 0);
        let mixed = string_concat(raw, b);
        assert_eq!(unsafe { string_header(mixed) }.flags & TEXT, 0);
        assert_eq!(string_hash(b), zrtl::fnv1a_bytes(&[b'x'; 40]) as i64);
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
