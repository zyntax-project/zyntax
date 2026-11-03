//! Patterns support.

use super::*;

pub(super) extern "C" fn host_pat_specials(pat: zrtl::StringConstPtr) -> bool {
    pattern::has_specials(unsafe { bytes_of(pat) })
}

/// The first match at or after 0-based `init`: its start, -1 or -2.
pub(super) extern "C" fn host_pat_find(
    s: zrtl::StringConstPtr,
    pat: zrtl::StringConstPtr,
    init: i64,
) -> i64 {
    let (src, pat) = unsafe { (bytes_of(s), bytes_of(pat)) };
    pattern::find(src, pat, init as usize)
}

/// A match exactly at 0-based `pos`: its end, -1 or -2.
pub(super) extern "C" fn host_pat_match_at(
    s: zrtl::StringConstPtr,
    pat: zrtl::StringConstPtr,
    pos: i64,
) -> i64 {
    let (src, pat) = unsafe { (bytes_of(s), bytes_of(pat)) };
    pattern::match_here(src, pat, pos as usize)
}

/// The end of the first match from 0-based `init` that does not end at
/// `last`: its end, -1 or -2.
pub(super) extern "C" fn host_pat_gmatch_next(
    s: zrtl::StringConstPtr,
    pat: zrtl::StringConstPtr,
    init: i64,
    last: i64,
) -> i64 {
    let (src, pat) = unsafe { (bytes_of(s), bytes_of(pat)) };
    pattern::gmatch_next(src, pat, init as usize, last)
}

pub(super) extern "C" fn host_pat_end() -> i64 {
    pattern::STATE.with(|st| st.borrow().end as i64)
}

pub(super) extern "C" fn host_pat_level() -> i64 {
    pattern::STATE.with(|st| st.borrow().level as i64)
}

/// Capture `i` of the last match: 0 for text, 1 for a position, -1
/// for an error (held for `$Lua$pat_error`).
pub(super) extern "C" fn host_pat_cap_kind(i: i64) -> i64 {
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Bytes(..)) => 0,
        Ok(pattern::CaptureValue::Position(_)) => 1,
        Err(e) => {
            pattern::STATE.with(|st| st.borrow_mut().error = e);
            -1
        }
    }
}

pub(super) extern "C" fn host_pat_cap_str(s: zrtl::StringConstPtr, i: i64) -> StringPtr {
    let src = unsafe { bytes_of(s) };
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Bytes(a, b)) => zrtl::string::string_from_bytes(&src[a..b]),
        _ => zrtl::string::string_from_bytes(b""),
    }
}

pub(super) extern "C" fn host_pat_cap_pos(i: i64) -> i64 {
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Position(p)) => p as i64,
        _ => 0,
    }
}

pub(super) extern "C" fn host_pat_error() -> StringPtr {
    pattern::STATE.with(|st| zrtl::string::string_from_bytes(st.borrow().error.as_bytes()))
}

pub(super) extern "C" fn host_buf_open() {
    BUFFERS.with(|b| b.borrow_mut().push(Vec::new()));
}

pub(super) extern "C" fn host_buf_push(s: zrtl::StringConstPtr) {
    let bytes = unsafe { bytes_of(s) };
    BUFFERS.with(|b| {
        if let Some(top) = b.borrow_mut().last_mut() {
            top.extend_from_slice(bytes);
        }
    });
}

/// Bytes `a..b` (0-based, `b` excluded) of `s`.
pub(super) extern "C" fn host_buf_push_range(s: zrtl::StringConstPtr, a: i64, b: i64) {
    let bytes = unsafe { bytes_of(s) };
    let (a, b) = (a.max(0) as usize, (b.max(0) as usize).min(bytes.len()));
    BUFFERS.with(|buf| {
        if let Some(top) = buf.borrow_mut().last_mut()
            && a < b
        {
            top.extend_from_slice(&bytes[a..b]);
        }
    });
}

/// The replacement string with `%n` expanded from the last match: 0,
/// or -2 with the error held.
pub(super) extern "C" fn host_buf_expand(
    s: zrtl::StringConstPtr,
    repl: zrtl::StringConstPtr,
) -> i64 {
    let (src, repl) = unsafe { (bytes_of(s), bytes_of(repl)) };
    let result = BUFFERS.with(|buf| {
        let mut buf = buf.borrow_mut();
        let Some(top) = buf.last_mut() else {
            return Ok(());
        };
        pattern::expand(src, repl, top)
    });
    match result {
        Ok(()) => 0,
        Err(e) => {
            pattern::STATE.with(|st| st.borrow_mut().error = e);
            -2
        }
    }
}

/// A module name as a path: dots become directory separators.
pub(super) extern "C" fn host_replace_dots(s: zrtl::StringConstPtr) -> StringPtr {
    let bytes: Vec<u8> = unsafe { bytes_of(s) }
        .iter()
        .map(|&b| if b == b'.' { b'/' } else { b })
        .collect();
    zrtl::string::string_from_bytes(&bytes)
}

pub(super) extern "C" fn host_buf_close() -> StringPtr {
    let bytes = BUFFERS.with(|b| b.borrow_mut().pop().unwrap_or_default());
    zrtl::string::string_from_bytes(&bytes)
}
