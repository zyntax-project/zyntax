//! Utf8 support.

use super::*;

pub(super) fn is_cont(b: u8) -> bool {
    b & 0xC0 == 0x80
}

/// Lua's `utf8_decode`: the code at `s[i..]` and the sequence's length,
/// or none for an invalid sequence.
pub fn utf8_decode(s: &[u8], i: usize, strict: bool) -> Option<(u32, usize)> {
    const LIMITS: [u32; 6] = [!0, 0x80, 0x800, 0x10000, 0x200000, 0x4000000];
    let c = *s.get(i)? as u32;
    if c < 0x80 {
        return Some((c, 1));
    }
    let mut res: u32 = 0;
    let mut count = 0;
    let mut lead = c;
    while lead & 0x40 != 0 {
        count += 1;
        let cc = *s.get(i + count).unwrap_or(&0) as u32;
        if cc & 0xC0 != 0x80 {
            return None;
        }
        res = (res << 6) | (cc & 0x3F);
        lead <<= 1;
    }
    // The lead byte as shifted by the loop, its length bits gone.
    res |= (lead & 0x7F) << (count * 5);
    if count > 5 || res > MAXUTF || res < LIMITS[count] {
        return None;
    }
    if strict && (res > MAXUNICODE || (0xD800..=0xDFFF).contains(&res)) {
        return None;
    }
    Some((res, count + 1))
}

/// Lua's `luaO_utf8esc`: a code as the bytes of its sequence.
pub fn utf8_encode(mut x: u32, out: &mut Vec<u8>) {
    if x < 0x80 {
        out.push(x as u8);
        return;
    }
    let mut buf = [0u8; 8];
    let mut n = 1;
    let mut mfb: u32 = 0x3f;
    loop {
        buf[8 - n] = (0x80 | (x & 0x3f)) as u8;
        n += 1;
        x >>= 6;
        mfb >>= 1;
        if x <= mfb {
            break;
        }
    }
    buf[8 - n] = ((!mfb << 1) | x) as u8;
    out.extend_from_slice(&buf[8 - n..]);
}

/// `u_posrelat`: a position, negative from the end, 0 when before it.
pub(super) fn u_posrelat(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if (pos.unsigned_abs() as usize) > len {
        0
    } else {
        len as i64 + pos + 1
    }
}

pub(super) extern "C" fn host_utf8_char(code: i64) -> StringPtr {
    let mut out = Vec::new();
    utf8_encode(code as u32, &mut out);
    zrtl::string::string_from_bytes(&out)
}

/// `utf8.len`: the count, -1 or -2 for a bad initial or final
/// position, or `-(pos) - 2` where decoding fails at 1-based `pos`.
pub(super) extern "C" fn host_utf8_len(s: zrtl::StringConstPtr, i: i64, j: i64, lax: bool) -> i64 {
    let s = unsafe { bytes_of(s) };
    let len = s.len() as i64;
    let mut posi = u_posrelat(i, s.len());
    let mut posj = u_posrelat(j, s.len());
    if !(1 <= posi && posi - 1 <= len) {
        return -1;
    }
    posi -= 1;
    posj -= 1;
    if posj >= len {
        return -2;
    }
    let mut n = 0;
    while posi <= posj {
        match utf8_decode(s, posi as usize, !lax) {
            Some((_, width)) => posi += width as i64,
            None => return -(posi + 1) - 2,
        }
        n += 1;
    }
    n
}

/// `utf8.offset`: the byte position, 0 for none, -1 for a position
/// out of bounds, -2 for one on a continuation byte.
pub(super) extern "C" fn host_utf8_offset(
    s: zrtl::StringConstPtr,
    n: i64,
    i: i64,
    has_i: bool,
) -> i64 {
    let s = unsafe { bytes_of(s) };
    let len = s.len() as i64;
    let default = if n >= 0 { 1 } else { len + 1 };
    let mut posi = u_posrelat(if has_i { i } else { default }, s.len());
    if !(1 <= posi && posi - 1 <= len) {
        return -1;
    }
    posi -= 1;
    let cont = |p: i64| (p as usize) < s.len() && is_cont(s[p as usize]);
    let mut n = n;
    if n == 0 {
        while posi > 0 && cont(posi) {
            posi -= 1;
        }
    } else {
        if cont(posi) {
            return -2;
        }
        if n < 0 {
            while n < 0 && posi > 0 {
                loop {
                    posi -= 1;
                    if !(posi > 0 && cont(posi)) {
                        break;
                    }
                }
                n += 1;
            }
        } else {
            n -= 1;
            while n > 0 && posi < len {
                loop {
                    posi += 1;
                    if !cont(posi) {
                        break;
                    }
                }
                n -= 1;
            }
        }
    }
    if n == 0 { posi + 1 } else { 0 }
}

/// `utf8.codepoint`: the codes between `i` and `j` held for
/// `$Lua$utf8_code_at`; their count, or -1 / -2 for a bad position,
/// -3 for an invalid sequence.
pub(super) extern "C" fn host_utf8_codepoint(
    s: zrtl::StringConstPtr,
    i: i64,
    j: i64,
    has_j: bool,
    lax: bool,
) -> i64 {
    let s = unsafe { bytes_of(s) };
    let posi = u_posrelat(i, s.len());
    let pose = u_posrelat(if has_j { j } else { posi }, s.len());
    if posi < 1 {
        return -1;
    }
    if pose > s.len() as i64 {
        return -2;
    }
    let mut codes = Vec::new();
    let mut at = (posi - 1) as usize;
    let end = pose as usize;
    while at < end {
        match utf8_decode(s, at, !lax) {
            Some((code, width)) => {
                codes.push(code);
                at += width;
            }
            None => return -3,
        }
    }
    let n = codes.len() as i64;
    UTF8_CODES.with(|c| *c.borrow_mut() = codes);
    n
}

pub(super) extern "C" fn host_utf8_code_at(k: i64) -> i64 {
    UTF8_CODES.with(|c| c.borrow().get(k as usize).copied().unwrap_or(0) as i64)
}

/// `utf8.codes`'s step: from the character at 1-based `n` (0 to
/// start), the next character's position with its code held, 0 at the
/// end, -1 for an invalid sequence.
pub(super) extern "C" fn host_utf8_next(s: zrtl::StringConstPtr, n: i64, lax: bool) -> i64 {
    let s = unsafe { bytes_of(s) };
    let len = s.len();
    let mut n = if n < 0 { len } else { n as usize };
    while n < len && is_cont(s[n]) {
        n += 1;
    }
    if n >= len {
        return 0;
    }
    match utf8_decode(s, n, !lax) {
        Some((code, width)) if !(n + width < len && is_cont(s[n + width])) => {
            UTF8_CODES.with(|c| *c.borrow_mut() = vec![code]);
            n as i64 + 1
        }
        _ => -1,
    }
}
