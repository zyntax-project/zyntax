//! Strings support.

use super::*;

/// `%.14g` of a float for `string.format`'s callers that have one.
pub(super) extern "C" fn host_format_g(x: f64, precision: i64) -> StringPtr {
    zrtl::string_new(&format_g(x, precision.clamp(1, 99) as usize))
}

/// A word as it is: declared by the library with another type at
/// either end, it reads an address back as the value it points to.
pub(super) extern "C" fn host_word(w: i64) -> i64 {
    w
}

pub(crate) unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    unsafe { zrtl::string_as_bytes(s) }
}

/// Lua's string positions: 1-based, negative from the end, clamped.
/// The range `i..=j` as byte offsets, empty when `i > j`.
pub(super) fn byte_range(len: usize, i: i64, j: i64) -> (usize, usize) {
    let len = len as i64;
    let start = if i < 0 {
        (len + i + 1).max(1)
    } else if i == 0 {
        1
    } else {
        i
    };
    let end = if j < 0 {
        len + j + 1
    } else if j > len {
        len
    } else {
        j
    };
    if start > end {
        (0, 0)
    } else {
        ((start - 1) as usize, end as usize)
    }
}

pub(super) extern "C" fn host_sub(s: zrtl::StringConstPtr, i: i64, j: i64) -> StringPtr {
    let bytes = unsafe { bytes_of(s) };
    let (a, b) = byte_range(bytes.len(), i, j);
    zrtl::string::string_from_bytes(&bytes[a..b])
}

/// The byte at 1-based position `i`, or -1 outside the string.
pub(super) extern "C" fn host_byte_at(s: zrtl::StringConstPtr, i: i64) -> i64 {
    let bytes = unsafe { bytes_of(s) };
    let (a, b) = byte_range(bytes.len(), i, i);
    if a < b { bytes[a] as i64 } else { -1 }
}

/// The bytes a string holds, or a foreign object is a buffer of, read in
/// place: what `string.len`, `sub`, `byte` and `unpack` read when their
/// subject is not a string by its static type.
pub(super) unsafe fn source_bytes<'a>(v: *const DynamicBox) -> &'a [u8] {
    if let Some(bytes) = unsafe { zyntax_embed::foreign::bytes_of(v as *mut DynamicBox) } {
        return bytes;
    }
    match unsafe { v.as_ref() } {
        Some(b) if b.tag.category() == TypeCategory::String => unsafe {
            bytes_of(b.data as zrtl::StringConstPtr)
        },
        _ => &[],
    }
}

pub(super) extern "C" fn host_len_any(v: *const DynamicBox) -> i64 {
    unsafe { source_bytes(v) }.len() as i64
}

pub(super) extern "C" fn host_sub_any(v: *const DynamicBox, i: i64, j: i64) -> StringPtr {
    let bytes = unsafe { source_bytes(v) };
    let (a, b) = byte_range(bytes.len(), i, j);
    zrtl::string::string_from_bytes(&bytes[a..b])
}

pub(super) extern "C" fn host_byte_at_any(v: *const DynamicBox, i: i64) -> i64 {
    let bytes = unsafe { source_bytes(v) };
    let (a, b) = byte_range(bytes.len(), i, i);
    if a < b { bytes[a] as i64 } else { -1 }
}

pub(super) extern "C" fn host_unpack_any(
    fmt: zrtl::StringConstPtr,
    v: *const DynamicBox,
    pos: i64,
) -> i64 {
    let (fmt, data) = unsafe { (bytes_of(fmt), source_bytes(v)) };
    match crate::pack::unpack(fmt, data, pos.max(0) as usize) {
        Ok((values, next)) => {
            let n = values.len() as i64;
            UNPACKED.with(|u| *u.borrow_mut() = (values, next));
            n
        }
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            -1
        }
    }
}

pub(super) extern "C" fn host_from_byte(b: i64) -> StringPtr {
    zrtl::string::string_from_bytes(&[b as u8])
}

pub(super) extern "C" fn host_reverse(s: zrtl::StringConstPtr) -> StringPtr {
    let mut bytes = unsafe { bytes_of(s) }.to_vec();
    bytes.reverse();
    zrtl::string::string_from_bytes(&bytes)
}

/// `string.rep`, or null for a result the reference bounds by a C
/// int: the library raises for it.
pub(super) extern "C" fn host_rep(
    s: zrtl::StringConstPtr,
    n: i64,
    sep: zrtl::StringConstPtr,
) -> StringPtr {
    let bytes = unsafe { bytes_of(s) };
    let sep = unsafe { bytes_of(sep) };
    if n > 0 && (bytes.len() + sep.len()) as u64 > i32::MAX as u64 / n as u64 {
        return std::ptr::null_mut();
    }
    let mut out = Vec::new();
    for k in 0..n.max(0) {
        if k > 0 {
            out.extend_from_slice(sep);
        }
        out.extend_from_slice(bytes);
    }
    zrtl::string::string_from_bytes(&out)
}

/// The 1-based position of `needle` in `s` at or after `init`, or 0.
pub(super) extern "C" fn host_find_plain(
    s: zrtl::StringConstPtr,
    needle: zrtl::StringConstPtr,
    init: i64,
) -> i64 {
    let hay = unsafe { bytes_of(s) };
    let needle = unsafe { bytes_of(needle) };
    // `init` is 1-based and at most one past the end, where only the
    // empty needle is found.
    let start = (init.max(1) - 1) as usize;
    if start > hay.len() {
        return 0;
    }
    if needle.is_empty() {
        return start as i64 + 1;
    }
    hay[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map_or(0, |p| (start + p) as i64 + 1)
}

/// A string from its bytes spelled as hex pairs: how a literal that is
/// not UTF-8 reaches the program.
pub(super) extern "C" fn host_bytes(hex: zrtl::StringConstPtr) -> StringPtr {
    let hex = unsafe { bytes_of(hex) };
    let bytes: Vec<u8> = hex
        .chunks(2)
        .filter_map(|pair| std::str::from_utf8(pair).ok())
        .filter_map(|pair| u8::from_str_radix(pair, 16).ok())
        .collect();
    zrtl::string::string_from_bytes(&bytes)
}

pub(super) extern "C" fn host_upper(s: zrtl::StringConstPtr) -> StringPtr {
    zrtl::string::string_from_bytes(&unsafe { bytes_of(s) }.to_ascii_uppercase())
}

pub(super) extern "C" fn host_lower(s: zrtl::StringConstPtr) -> StringPtr {
    zrtl::string::string_from_bytes(&unsafe { bytes_of(s) }.to_ascii_lowercase())
}
