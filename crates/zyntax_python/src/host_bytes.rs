//! The host side of the `bytes` methods: the search, edit, case and
//! split operations of Python's `bytes`. Every position is a byte
//! offset, case and character classes are ASCII only, and the contents
//! need not be UTF-8.

use zrtl::{StringConstPtr, StringPtr, bytes_new, string_as_bytes};

/// Whitespace to `bytes.strip`, `split` and `isspace`.
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// The window `start..end` of `len` bytes as a slice clamps it; `end` is
/// `i64::MAX` when it was not given. `None` when the window is past the
/// end or empty with a start beyond it.
fn window(len: usize, start: i64, end: i64) -> Option<(usize, usize)> {
    let len = len as i64;
    let start = if start < 0 {
        (start + len).max(0)
    } else {
        start
    };
    let end = if end < 0 {
        (end + len).max(0)
    } else {
        end.min(len)
    };
    if start > len || end < start {
        return None;
    }
    Some((start as usize, end as usize))
}

fn find_in(h: &[u8], n: &[u8]) -> Option<usize> {
    match n.len() {
        0 => Some(0),
        1 => h.iter().position(|&b| b == n[0]),
        w => h.windows(w).position(|x| x == n),
    }
}

fn rfind_in(h: &[u8], n: &[u8]) -> Option<usize> {
    match n.len() {
        0 => Some(h.len()),
        1 => h.iter().rposition(|&b| b == n[0]),
        w => h.windows(w).rposition(|x| x == n),
    }
}

/// Matches of `n` in `h` that do not overlap, as `bytes.count` counts.
fn count_in(h: &[u8], n: &[u8]) -> i64 {
    if n.is_empty() {
        return h.len() as i64 + 1;
    }
    let mut count = 0;
    let mut at = 0;
    while let Some(i) = find_in(&h[at..], n) {
        count += 1;
        at += i + n.len();
    }
    count
}

/// `find` (mode 0), `rfind` (1) and `count` (2) over the window.
pub fn find_range(s: &[u8], sub: &[u8], start: i64, end: i64, mode: i64) -> i64 {
    let missing = if mode == 2 { 0 } else { -1 };
    let Some((a, b)) = window(s.len(), start, end) else {
        return missing;
    };
    let h = &s[a..b];
    match mode {
        2 => count_in(h, sub),
        0 => find_in(h, sub).map_or(-1, |i| (a + i) as i64),
        _ => rfind_in(h, sub).map_or(-1, |i| (a + i) as i64),
    }
}

/// `startswith` (`ends` 0) and `endswith` (1) over the window.
pub fn edge_range(s: &[u8], affix: &[u8], start: i64, end: i64, ends: i64) -> bool {
    let Some((a, b)) = window(s.len(), start, end) else {
        return false;
    };
    let h = &s[a..b];
    if ends == 0 {
        h.starts_with(affix)
    } else {
        h.ends_with(affix)
    }
}

/// `replace`: the first `count` matches, every one when `count` is
/// negative. An empty `from` matches before each byte and at the end.
pub fn replace_n(s: &[u8], from: &[u8], to: &[u8], count: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut left = if count < 0 { i64::MAX } else { count };
    if from.is_empty() {
        for &b in s {
            if left > 0 {
                out.extend_from_slice(to);
                left -= 1;
            }
            out.push(b);
        }
        if left > 0 {
            out.extend_from_slice(to);
        }
        return out;
    }
    let mut at = 0;
    while left > 0 {
        match find_in(&s[at..], from) {
            Some(i) => {
                out.extend_from_slice(&s[at..at + i]);
                out.extend_from_slice(to);
                at += i + from.len();
                left -= 1;
            }
            None => break,
        }
    }
    out.extend_from_slice(&s[at..]);
    out
}

/// `upper` (0), `lower` (1), `swapcase` (2), `capitalize` (3) and
/// `title` (4), over ASCII letters.
pub fn case_map(s: &[u8], kind: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut after_letter = false;
    for (i, &b) in s.iter().enumerate() {
        out.push(match kind {
            0 => b.to_ascii_uppercase(),
            1 => b.to_ascii_lowercase(),
            2 if b.is_ascii_lowercase() => b.to_ascii_uppercase(),
            2 => b.to_ascii_lowercase(),
            3 if i == 0 => b.to_ascii_uppercase(),
            3 => b.to_ascii_lowercase(),
            _ if after_letter => b.to_ascii_lowercase(),
            _ => b.to_ascii_uppercase(),
        });
        after_letter = b.is_ascii_alphabetic();
    }
    out
}

/// `isalpha` (0), `isalnum` (1), `isdigit` (2), `isspace` (3),
/// `isupper` (4), `islower` (5), `istitle` (6) and `isascii` (7).
pub fn check(s: &[u8], kind: i64) -> bool {
    let all = |f: fn(&u8) -> bool| !s.is_empty() && s.iter().all(f);
    match kind {
        0 => all(u8::is_ascii_alphabetic),
        1 => all(u8::is_ascii_alphanumeric),
        2 => all(u8::is_ascii_digit),
        3 => all(|b| is_space(*b)),
        4 => s.iter().any(u8::is_ascii_uppercase) && !s.iter().any(u8::is_ascii_lowercase),
        5 => s.iter().any(u8::is_ascii_lowercase) && !s.iter().any(u8::is_ascii_uppercase),
        6 => {
            let mut previous_cased = false;
            let mut cased = false;
            for &b in s {
                if b.is_ascii_uppercase() {
                    if previous_cased {
                        return false;
                    }
                    previous_cased = true;
                    cased = true;
                } else if b.is_ascii_lowercase() {
                    if !previous_cased {
                        return false;
                    }
                    cased = true;
                } else {
                    previous_cased = false;
                }
            }
            cased
        }
        _ => s.is_ascii(),
    }
}

/// `strip` (`mode` 3), `lstrip` (1) and `rstrip` (2) of the bytes in
/// `chars`, or of whitespace when `by_space`.
pub fn strip<'a>(s: &'a [u8], chars: &[u8], mode: i64, by_space: bool) -> &'a [u8] {
    let drop = |b: &u8| {
        if by_space {
            is_space(*b)
        } else {
            chars.contains(b)
        }
    };
    let mut s = s;
    if mode & 1 != 0 {
        while let [first, rest @ ..] = s {
            if !drop(first) {
                break;
            }
            s = rest;
        }
    }
    if mode & 2 != 0 {
        while let [rest @ .., last] = s {
            if !drop(last) {
                break;
            }
            s = rest;
        }
    }
    s
}

/// `ljust` (`mode` 0), `rjust` (1), `center` (2) and `zfill` (3) to
/// `width` bytes.
pub fn justify(s: &[u8], width: i64, fill: u8, mode: i64) -> Vec<u8> {
    let len = s.len() as i64;
    if width <= len {
        return s.to_vec();
    }
    let pad = (width - len) as usize;
    let mut out = Vec::with_capacity(width as usize);
    match mode {
        0 => {
            out.extend_from_slice(s);
            out.resize(width as usize, fill);
        }
        1 => {
            out.resize(pad, fill);
            out.extend_from_slice(s);
        }
        2 => {
            // The odd fill goes on the left when the width is odd.
            let left = pad / 2 + (pad & width as usize & 1);
            out.resize(left, fill);
            out.extend_from_slice(s);
            out.resize(width as usize, fill);
        }
        _ => {
            let sign = matches!(s.first(), Some(b'+' | b'-'));
            if sign {
                out.push(s[0]);
            }
            out.resize(out.len() + pad, b'0');
            out.extend_from_slice(&s[usize::from(sign)..]);
        }
    }
    out
}

/// The pieces `split` (mode 0), `rsplit` (1), the whitespace forms of
/// both (2, 3), `splitlines` (4, `limit` is whether to keep the line
/// ends), `partition` (5) and `rpartition` (6) cut `s` into, as
/// `(start, end)` offsets in order. `limit` is the most cuts, negative
/// for no limit.
pub fn cuts(s: &[u8], sep: &[u8], limit: i64, mode: i64) -> Vec<(usize, usize)> {
    let n = s.len();
    let mut out = Vec::new();
    match mode {
        0 => {
            let mut at = 0;
            let mut made = 0;
            while limit < 0 || made < limit {
                match find_in(&s[at..], sep) {
                    Some(i) => {
                        out.push((at, at + i));
                        at += i + sep.len();
                        made += 1;
                    }
                    None => break,
                }
            }
            out.push((at, n));
        }
        1 => {
            let mut end = n;
            let mut made = 0;
            while limit < 0 || made < limit {
                match rfind_in(&s[..end], sep) {
                    Some(i) => {
                        out.push((i + sep.len(), end));
                        end = i;
                        made += 1;
                    }
                    None => break,
                }
            }
            out.push((0, end));
            out.reverse();
        }
        2 => {
            let mut at = 0;
            let mut made = 0;
            loop {
                while at < n && is_space(s[at]) {
                    at += 1;
                }
                if at >= n {
                    break;
                }
                if limit >= 0 && made >= limit {
                    out.push((at, n));
                    break;
                }
                let start = at;
                while at < n && !is_space(s[at]) {
                    at += 1;
                }
                out.push((start, at));
                made += 1;
            }
        }
        3 => {
            let mut end = n;
            let mut made = 0;
            loop {
                while end > 0 && is_space(s[end - 1]) {
                    end -= 1;
                }
                if end == 0 {
                    break;
                }
                if limit >= 0 && made >= limit {
                    out.push((0, end));
                    break;
                }
                let stop = end;
                while end > 0 && !is_space(s[end - 1]) {
                    end -= 1;
                }
                out.push((end, stop));
                made += 1;
            }
            out.reverse();
        }
        4 => {
            let mut at = 0;
            while at < n {
                let mut cut = at;
                while cut < n && s[cut] != b'\n' && s[cut] != b'\r' {
                    cut += 1;
                }
                let mut after = cut;
                if cut < n {
                    after += if s[cut] == b'\r' && s.get(cut + 1) == Some(&b'\n') {
                        2
                    } else {
                        1
                    };
                }
                out.push((at, if limit != 0 { after } else { cut }));
                at = after;
            }
        }
        _ => {
            let found = if mode == 5 {
                find_in(s, sep)
            } else {
                rfind_in(s, sep)
            };
            match found {
                Some(i) => {
                    out.push((0, i));
                    out.push((i, i + sep.len()));
                    out.push((i + sep.len(), n));
                }
                None if mode == 5 => out.extend([(0, n), (n, n), (n, n)]),
                None => out.extend([(0, 0), (0, 0), (0, n)]),
            }
        }
    }
    out
}

/// `expandtabs`: tabs become spaces up to the next multiple of `size`
/// columns, which a line end resets.
pub fn expand_tabs(s: &[u8], size: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut column = 0usize;
    for &b in s {
        match b {
            b'\t' if size > 0 => {
                let pad = size as usize - column % size as usize;
                out.resize(out.len() + pad, b' ');
                column += pad;
            }
            b'\t' => {}
            b'\n' | b'\r' => {
                out.push(b);
                column = 0;
            }
            _ => {
                out.push(b);
                column += 1;
            }
        }
    }
    out
}

/// `translate`: the bytes of `delete` dropped, the rest mapped through
/// `table` when there is one.
pub fn translate(s: &[u8], table: &[u8], delete: &[u8]) -> Vec<u8> {
    s.iter()
        .filter(|b| !delete.contains(b))
        .map(|&b| table.get(b as usize).copied().unwrap_or(b))
        .collect()
}

/// `bytes.maketrans`: the 256-byte table taking each byte of `from` to
/// the byte of `to` at its position.
pub fn maketrans(from: &[u8], to: &[u8]) -> Vec<u8> {
    let mut table: Vec<u8> = (0..=255).collect();
    for (&f, &t) in from.iter().zip(to) {
        table[f as usize] = t;
    }
    table
}

/// `bytes.find` (`mode` 0), `rfind` (1) and `count` (2) over a window.
pub(crate) extern "C" fn host_bytes_find(
    s: StringConstPtr,
    sub: StringConstPtr,
    start: i64,
    end: i64,
    mode: i64,
) -> i64 {
    let (s, sub) = unsafe { (string_as_bytes(s), string_as_bytes(sub)) };
    find_range(s, sub, start, end, mode)
}

/// `bytes.startswith` (`ends` 0) and `endswith` (1) over a window.
pub(crate) extern "C" fn host_bytes_edge(
    s: StringConstPtr,
    affix: StringConstPtr,
    start: i64,
    end: i64,
    ends: i64,
) -> i32 {
    let (s, affix) = unsafe { (string_as_bytes(s), string_as_bytes(affix)) };
    edge_range(s, affix, start, end, ends) as i32
}

/// `bytes.replace(from, to, count)`.
pub(crate) extern "C" fn host_bytes_replace_n(
    s: StringConstPtr,
    from: StringConstPtr,
    to: StringConstPtr,
    count: i64,
) -> StringPtr {
    let (s, from, to) = unsafe {
        (
            string_as_bytes(s),
            string_as_bytes(from),
            string_as_bytes(to),
        )
    };
    bytes_new(&replace_n(s, from, to, count))
}

/// The ASCII case maps of `bytes`; `kind` picks one.
pub(crate) extern "C" fn host_bytes_case_map(s: StringConstPtr, kind: i64) -> StringPtr {
    bytes_new(&case_map(unsafe { string_as_bytes(s) }, kind))
}

/// The `bytes.is*()` predicates; `kind` picks one.
pub(crate) extern "C" fn host_bytes_check(s: StringConstPtr, kind: i64) -> i32 {
    check(unsafe { string_as_bytes(s) }, kind) as i32
}

/// `bytes.strip`, `lstrip` and `rstrip`; `mode` bit 0 strips the left,
/// bit 1 the right, bit 2 says `chars` was not given, so whitespace goes.
pub(crate) extern "C" fn host_bytes_strip(
    s: StringConstPtr,
    chars: StringConstPtr,
    mode: i64,
) -> StringPtr {
    let (s, chars) = unsafe { (string_as_bytes(s), string_as_bytes(chars)) };
    bytes_new(strip(s, chars, mode & 3, mode & 4 != 0))
}

/// `bytes.ljust`, `rjust`, `center` and `zfill`; `fill` is one byte.
pub(crate) extern "C" fn host_bytes_justify(
    s: StringConstPtr,
    width: i64,
    fill: StringConstPtr,
    mode: i64,
) -> StringPtr {
    let (s, fill) = unsafe { (string_as_bytes(s), string_as_bytes(fill)) };
    bytes_new(&justify(
        s,
        width,
        fill.first().copied().unwrap_or(b' '),
        mode,
    ))
}

/// The cut offsets of `cuts`, as a byte string of little-endian `i64`
/// pairs, which `host_bytes_cut_at` reads back.
pub(crate) extern "C" fn host_bytes_cuts(
    s: StringConstPtr,
    sep: StringConstPtr,
    limit: i64,
    mode: i64,
) -> StringPtr {
    let (s, sep) = unsafe { (string_as_bytes(s), string_as_bytes(sep)) };
    let mut packed = Vec::new();
    for (a, b) in cuts(s, sep, limit, mode) {
        packed.extend_from_slice(&(a as i64).to_le_bytes());
        packed.extend_from_slice(&(b as i64).to_le_bytes());
    }
    bytes_new(&packed)
}

/// The `k`th `i64` of a string `host_bytes_cuts` made.
pub(crate) extern "C" fn host_bytes_cut_at(packed: StringConstPtr, k: i64) -> i64 {
    let packed = unsafe { string_as_bytes(packed) };
    let at = k as usize * 8;
    match packed.get(at..at + 8) {
        Some(b) => i64::from_le_bytes(b.try_into().expect("eight bytes")),
        None => 0,
    }
}

/// `bytes.expandtabs(size)`.
pub(crate) extern "C" fn host_bytes_expandtabs(s: StringConstPtr, size: i64) -> StringPtr {
    bytes_new(&expand_tabs(unsafe { string_as_bytes(s) }, size))
}

/// `bytes.translate(table, delete)`; `table` is empty for none.
pub(crate) extern "C" fn host_bytes_translate(
    s: StringConstPtr,
    table: StringConstPtr,
    delete: StringConstPtr,
) -> StringPtr {
    let (s, table, delete) = unsafe {
        (
            string_as_bytes(s),
            string_as_bytes(table),
            string_as_bytes(delete),
        )
    };
    bytes_new(&translate(s, table, delete))
}

/// `bytes.maketrans(from, to)`.
pub(crate) extern "C" fn host_bytes_maketrans(
    from: StringConstPtr,
    to: StringConstPtr,
) -> StringPtr {
    let (from, to) = unsafe { (string_as_bytes(from), string_as_bytes(to)) };
    bytes_new(&maketrans(from, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces<'a>(s: &'a [u8], sep: &[u8], limit: i64, mode: i64) -> Vec<&'a [u8]> {
        cuts(s, sep, limit, mode)
            .into_iter()
            .map(|(a, b)| &s[a..b])
            .collect()
    }

    #[test]
    fn find_windows_and_counts() {
        assert_eq!(find_range(b"abcabc", b"bc", 0, i64::MAX, 0), 1);
        assert_eq!(find_range(b"abcabc", b"bc", 0, i64::MAX, 1), 4);
        assert_eq!(find_range(b"abcabc", b"bc", 2, i64::MAX, 0), 4);
        assert_eq!(find_range(b"abcabc", b"bc", -2, i64::MAX, 0), 4);
        assert_eq!(find_range(b"aaaa", b"aa", 0, i64::MAX, 2), 2);
        assert_eq!(find_range(b"abc", b"", 1, i64::MAX, 0), 1);
        assert_eq!(find_range(b"abc", b"", 4, i64::MAX, 0), -1);
        assert_eq!(find_range(b"abc", b"", 0, i64::MAX, 1), 3);
        assert_eq!(find_range(b"abc", b"", 0, i64::MAX, 2), 4);
        // Bytes that are not UTF-8, and one that cuts a character.
        assert_eq!(find_range(b"\x80\x80ab", b"\x80", 1, i64::MAX, 0), 1);
        assert_eq!(find_range("é".as_bytes(), b"\xa9", 0, i64::MAX, 0), 1);
    }

    #[test]
    fn edges_follow_the_window() {
        assert!(edge_range(b"hello", b"he", 0, i64::MAX, 0));
        assert!(!edge_range(b"hello", b"he", 1, i64::MAX, 0));
        assert!(edge_range(b"hello", b"lo", 0, i64::MAX, 1));
        assert!(edge_range(b"hello", b"el", 1, 3, 0));
        assert!(!edge_range(b"hello", b"", 9, i64::MAX, 0));
    }

    #[test]
    fn replace_counts_and_empty_patterns() {
        assert_eq!(replace_n(b"a-b-c", b"-", b"+", -1), b"a+b+c");
        assert_eq!(replace_n(b"a-b-c", b"-", b"+", 1), b"a+b-c");
        assert_eq!(replace_n(b"abc", b"", b"-", -1), b"-a-b-c-");
        assert_eq!(replace_n(b"abc", b"", b"-", 2), b"-a-bc");
        assert_eq!(replace_n(b"abc", b"x", b"-", -1), b"abc");
    }

    #[test]
    fn case_is_ascii_only() {
        assert_eq!(case_map(b"aB\xe9", 0), b"AB\xe9");
        assert_eq!(case_map(b"aB1c", 2), b"Ab1C");
        assert_eq!(case_map(b"hELLO wORLD", 3), b"Hello world");
        assert_eq!(case_map(b"hello wORLD-x 1a", 4), b"Hello World-X 1A");
    }

    #[test]
    fn checks_are_ascii_only() {
        assert!(check(b"abc", 0));
        assert!(!check(b"", 0));
        assert!(!check(b"ab\xe9", 0));
        assert!(check(b"a1", 1));
        assert!(check(b"123", 2));
        assert!(check(b" \t\n\x0b\x0c\r", 3));
        assert!(!check(b"\x1c", 3));
        assert!(check(b"AB1", 4));
        assert!(!check(b"1", 4));
        assert!(check(b"ab1", 5));
        assert!(check(b"Hello World", 6));
        assert!(!check(b"Hello world", 6));
        assert!(check(b"", 7));
        assert!(!check(b"\x80", 7));
    }

    #[test]
    fn strip_by_space_or_set() {
        assert_eq!(strip(b"  ab \n", b"", 3, true), b"ab");
        assert_eq!(strip(b"  ab \n", b"", 1, true), b"ab \n");
        assert_eq!(strip(b"  ab \n", b"", 2, true), b"  ab");
        assert_eq!(strip(b"xxabyx", b"xy", 3, false), b"ab");
        assert_eq!(strip(b"\x1cab", b"", 3, true), b"\x1cab");
    }

    #[test]
    fn justify_pads_as_python_does() {
        assert_eq!(justify(b"ab", 5, b'.', 0), b"ab...");
        assert_eq!(justify(b"ab", 5, b'.', 1), b"...ab");
        assert_eq!(justify(b"ab", 5, b'.', 2), b"..ab.");
        assert_eq!(justify(b"ab", 4, b'.', 2), b".ab.");
        assert_eq!(justify(b"-42", 6, b' ', 3), b"-00042");
        assert_eq!(justify(b"abc", 2, b'.', 0), b"abc");
    }

    #[test]
    fn splits_by_separator() {
        assert_eq!(pieces(b"a,b,,c", b",", -1, 0), [&b"a"[..], b"b", b"", b"c"]);
        assert_eq!(pieces(b"a,b,c", b",", 1, 0), [&b"a"[..], b"b,c"]);
        assert_eq!(pieces(b"a,b,c", b",", 1, 1), [&b"a,b"[..], b"c"]);
        assert_eq!(pieces(b"", b",", -1, 0), [&b""[..]]);
        assert_eq!(pieces(b"a--b", b"--", -1, 0), [&b"a"[..], b"b"]);
    }

    #[test]
    fn splits_by_whitespace() {
        assert_eq!(pieces(b"  a  b c ", b"", -1, 2), [&b"a"[..], b"b", b"c"]);
        assert_eq!(pieces(b"  a  b c ", b"", 1, 2), [&b"a"[..], b"b c "]);
        assert_eq!(pieces(b"  a  b c ", b"", 1, 3), [&b"  a  b"[..], b"c"]);
        assert_eq!(pieces(b"   ", b"", -1, 2), Vec::<&[u8]>::new());
    }

    #[test]
    fn splits_lines_at_newlines_only() {
        assert_eq!(
            pieces(b"a\nb\r\nc\rd\x0be", b"", 0, 4),
            [&b"a"[..], b"b", b"c", b"d\x0be"]
        );
        assert_eq!(pieces(b"a\nb\r\n", b"", 1, 4), [&b"a\n"[..], b"b\r\n"]);
        assert_eq!(pieces(b"", b"", 0, 4), Vec::<&[u8]>::new());
    }

    #[test]
    fn partitions_into_three() {
        assert_eq!(pieces(b"a=b=c", b"=", 0, 5), [&b"a"[..], b"=", b"b=c"]);
        assert_eq!(pieces(b"a=b=c", b"=", 0, 6), [&b"a=b"[..], b"=", b"c"]);
        assert_eq!(pieces(b"abc", b"=", 0, 5), [&b"abc"[..], b"", b""]);
        assert_eq!(pieces(b"abc", b"=", 0, 6), [&b""[..], b"", b"abc"]);
    }

    #[test]
    fn cut_offsets_round_trip() {
        let s = bytes_new(b"a,b");
        let sep = bytes_new(b",");
        let packed = host_bytes_cuts(s, sep, -1, 0);
        assert_eq!(host_bytes_cut_at(packed, 0), 0);
        assert_eq!(host_bytes_cut_at(packed, 1), 1);
        assert_eq!(host_bytes_cut_at(packed, 2), 2);
        assert_eq!(host_bytes_cut_at(packed, 3), 3);
        assert_eq!(host_bytes_cut_at(packed, 4), 0);
    }
    #[test]
    fn tabs_expand_to_columns() {
        assert_eq!(expand_tabs(b"a\tb", 4), b"a   b");
        assert_eq!(expand_tabs(b"ab\tc\n\td", 4), b"ab  c\n    d");
        assert_eq!(expand_tabs(b"a\tb", 0), b"ab");
    }

    #[test]
    fn translation_maps_and_deletes() {
        let table = maketrans(b"abc", b"xyz");
        assert_eq!(table.len(), 256);
        assert_eq!(translate(b"aabbcd", &table, b""), b"xxyyzd");
        assert_eq!(translate(b"aabbcd", &table, b"d"), b"xxyyz");
        assert_eq!(translate(b"abc", b"", b"b"), b"ac");
    }
}
