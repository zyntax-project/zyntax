//! Lua patterns, matched as the reference does: a backtracking
//! matcher over bytes with classes, sets, the four quantifiers,
//! anchors, `%b`, `%f` and up to 32 captures, position captures
//! included. A match's captures stay in a per-thread state the
//! library reads back one by one.

use std::cell::RefCell;

pub const MAX_CAPTURES: usize = 32;
const MAX_DEPTH: usize = 200;

const CAP_UNFINISHED: isize = -1;
const CAP_POSITION: isize = -2;

const ESC: u8 = b'%';
const SPECIALS: &[u8] = b"^$*+?.([%-";

/// A capture: where it starts and how long it is, or one of the two
/// markers while unfinished or for a position capture.
#[derive(Clone, Copy)]
pub struct Capture {
    pub start: usize,
    pub len: isize,
}

pub struct Matcher<'a> {
    src: &'a [u8],
    pat: &'a [u8],
    depth: usize,
    pub level: usize,
    pub capture: [Capture; MAX_CAPTURES],
}

type MatchResult = Result<Option<usize>, String>;

/// Whether any byte of the pattern has a meaning.
pub fn has_specials(pat: &[u8]) -> bool {
    pat.iter().any(|b| SPECIALS.contains(b))
}

fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}
fn is_cntrl(c: u8) -> bool {
    c < 32 || c == 127
}
fn is_graph(c: u8) -> bool {
    (33..=126).contains(&c)
}
fn is_punct(c: u8) -> bool {
    is_graph(c) && !c.is_ascii_alphanumeric()
}
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `%a`, `%d` and the rest; an upper-case letter is the complement.
fn match_class(c: u8, class: u8) -> bool {
    let res = match class.to_ascii_lowercase() {
        b'a' => is_alpha(c),
        b'c' => is_cntrl(c),
        b'd' => c.is_ascii_digit(),
        b'g' => is_graph(c),
        b'l' => c.is_ascii_lowercase(),
        b'p' => is_punct(c),
        b's' => is_space(c),
        b'u' => c.is_ascii_uppercase(),
        b'w' => c.is_ascii_alphanumeric(),
        b'x' => c.is_ascii_hexdigit(),
        b'z' => c == 0,
        _ => return class == c,
    };
    if class.is_ascii_uppercase() {
        !res
    } else {
        res
    }
}

impl<'a> Matcher<'a> {
    pub fn new(src: &'a [u8], pat: &'a [u8]) -> Self {
        Matcher {
            src,
            pat,
            depth: MAX_DEPTH,
            level: 0,
            capture: [Capture { start: 0, len: 0 }; MAX_CAPTURES],
        }
    }

    pub fn reset(&mut self) {
        self.level = 0;
        self.depth = MAX_DEPTH;
    }

    /// Where the item starting at `p` ends: past its class or set.
    fn class_end(&self, mut p: usize) -> Result<usize, String> {
        let c = self.pat[p];
        p += 1;
        if c == ESC {
            if p >= self.pat.len() {
                return Err("malformed pattern (ends with '%')".into());
            }
            return Ok(p + 1);
        }
        if c == b'[' {
            if p < self.pat.len() && self.pat[p] == b'^' {
                p += 1;
            }
            // One byte is taken before the first look for `]`, so a
            // set may start with a literal one.
            loop {
                if p >= self.pat.len() {
                    return Err("malformed pattern (missing ']')".into());
                }
                let cc = self.pat[p];
                p += 1;
                if cc == ESC && p < self.pat.len() {
                    p += 1;
                }
                if p < self.pat.len() && self.pat[p] == b']' {
                    return Ok(p + 1);
                }
            }
        }
        Ok(p)
    }

    /// `[set]` from `p` (the `[`) to `ec` (the `]`): whether `c` is in it.
    fn match_bracket_class(&self, c: u8, mut p: usize, ec: usize) -> bool {
        let mut sig = true;
        p += 1;
        if self.pat[p] == b'^' {
            sig = false;
            p += 1;
        }
        while p < ec {
            if self.pat[p] == ESC {
                p += 1;
                if match_class(c, self.pat[p]) {
                    return sig;
                }
                p += 1;
            } else if p + 2 < ec && self.pat[p + 1] == b'-' {
                if self.pat[p] <= c && c <= self.pat[p + 2] {
                    return sig;
                }
                p += 3;
            } else {
                if self.pat[p] == c {
                    return sig;
                }
                p += 1;
            }
        }
        !sig
    }

    /// Whether the byte at `s` matches the single item `p..ep`.
    fn single_match(&self, s: usize, p: usize, ep: usize) -> bool {
        if s >= self.src.len() {
            return false;
        }
        let c = self.src[s];
        match self.pat[p] {
            b'.' => true,
            ESC => match_class(c, self.pat[p + 1]),
            b'[' => self.match_bracket_class(c, p, ep - 1),
            pc => pc == c,
        }
    }

    fn match_balance(&self, s: usize, p: usize) -> Result<Option<usize>, String> {
        if p + 1 >= self.pat.len() {
            return Err("malformed pattern (missing arguments to '%b')".into());
        }
        if s >= self.src.len() || self.src[s] != self.pat[p] {
            return Ok(None);
        }
        let (b, e) = (self.pat[p], self.pat[p + 1]);
        let mut cont = 1;
        let mut s = s + 1;
        while s < self.src.len() {
            let c = self.src[s];
            if c == e {
                cont -= 1;
                if cont == 0 {
                    return Ok(Some(s + 1));
                }
            } else if c == b {
                cont += 1;
            }
            s += 1;
        }
        Ok(None)
    }

    fn max_expand(&mut self, s: usize, p: usize, ep: usize) -> MatchResult {
        let mut i = 0;
        while self.single_match(s + i, p, ep) {
            i += 1;
        }
        // The longest run first, then shorter ones.
        loop {
            if let Some(res) = self.do_match(s + i, ep + 1)? {
                return Ok(Some(res));
            }
            if i == 0 {
                return Ok(None);
            }
            i -= 1;
        }
    }

    fn min_expand(&mut self, mut s: usize, p: usize, ep: usize) -> MatchResult {
        loop {
            if let Some(res) = self.do_match(s, ep + 1)? {
                return Ok(Some(res));
            }
            if self.single_match(s, p, ep) {
                s += 1;
            } else {
                return Ok(None);
            }
        }
    }

    fn start_capture(&mut self, s: usize, p: usize, what: isize) -> MatchResult {
        if self.level >= MAX_CAPTURES {
            return Err("too many captures".into());
        }
        self.capture[self.level] = Capture {
            start: s,
            len: what,
        };
        self.level += 1;
        let res = self.do_match(s, p)?;
        if res.is_none() {
            self.level -= 1;
        }
        Ok(res)
    }

    fn end_capture(&mut self, s: usize, p: usize) -> MatchResult {
        let l = self.capture_to_close()?;
        self.capture[l].len = (s - self.capture[l].start) as isize;
        let res = self.do_match(s, p)?;
        if res.is_none() {
            self.capture[l].len = CAP_UNFINISHED;
        }
        Ok(res)
    }

    fn capture_to_close(&self) -> Result<usize, String> {
        let mut level = self.level;
        while level > 0 {
            level -= 1;
            if self.capture[level].len == CAP_UNFINISHED {
                return Ok(level);
            }
        }
        Err("invalid pattern capture".into())
    }

    /// `%1`..`%9`: the capture's text again.
    fn match_capture(&self, s: usize, l: u8) -> Result<Option<usize>, String> {
        let l = self.check_capture(l)?;
        let cap = self.capture[l];
        let len = cap.len as usize;
        if self.src.len() - s >= len && self.src[cap.start..cap.start + len] == self.src[s..s + len]
        {
            Ok(Some(s + len))
        } else {
            Ok(None)
        }
    }

    fn check_capture(&self, l: u8) -> Result<usize, String> {
        let l = l as isize - b'1' as isize;
        if l < 0 || l as usize >= self.level || self.capture[l as usize].len == CAP_UNFINISHED {
            return Err(format!("invalid capture index %{} in pattern", l + 1));
        }
        Ok(l as usize)
    }

    /// The end of a match of `pat[p..]` at `s`, or none.
    fn do_match(&mut self, mut s: usize, mut p: usize) -> MatchResult {
        if self.depth == 0 {
            return Err("pattern too complex".into());
        }
        self.depth -= 1;
        let result = 'outer: loop {
            if p >= self.pat.len() {
                break Some(s);
            }
            match self.pat[p] {
                b'(' => {
                    let res = if p + 1 < self.pat.len() && self.pat[p + 1] == b')' {
                        self.start_capture(s, p + 2, CAP_POSITION)?
                    } else {
                        self.start_capture(s, p + 1, CAP_UNFINISHED)?
                    };
                    break res;
                }
                b')' => break self.end_capture(s, p + 1)?,
                b'$' if p + 1 == self.pat.len() => {
                    break if s == self.src.len() { Some(s) } else { None };
                }
                ESC if p + 1 < self.pat.len() => match self.pat[p + 1] {
                    b'b' => match self.match_balance(s, p + 2)? {
                        Some(next) => {
                            s = next;
                            p += 4;
                            continue 'outer;
                        }
                        None => break None,
                    },
                    b'f' => {
                        p += 2;
                        if p >= self.pat.len() || self.pat[p] != b'[' {
                            return Err("missing '[' after '%f' in pattern".into());
                        }
                        let ep = self.class_end(p)?;
                        let previous = if s == 0 { 0 } else { self.src[s - 1] };
                        let current = if s < self.src.len() { self.src[s] } else { 0 };
                        if !self.match_bracket_class(previous, p, ep - 1)
                            && self.match_bracket_class(current, p, ep - 1)
                        {
                            p = ep;
                            continue 'outer;
                        }
                        break None;
                    }
                    d @ b'0'..=b'9' => match self.match_capture(s, d)? {
                        Some(next) => {
                            s = next;
                            p += 2;
                            continue 'outer;
                        }
                        None => break None,
                    },
                    _ => {}
                },
                _ => {}
            }
            // A single item, with its quantifier.
            let ep = self.class_end(p)?;
            let epc = if ep < self.pat.len() { self.pat[ep] } else { 0 };
            if !self.single_match(s, p, ep) {
                if epc == b'*' || epc == b'?' || epc == b'-' {
                    p = ep + 1;
                    continue 'outer;
                }
                break None;
            }
            match epc {
                b'?' => {
                    if let Some(res) = self.do_match(s + 1, ep + 1)? {
                        break Some(res);
                    }
                    p = ep + 1;
                    continue 'outer;
                }
                b'+' => break self.max_expand(s + 1, p, ep)?,
                b'*' => break self.max_expand(s, p, ep)?,
                b'-' => break self.min_expand(s, p, ep)?,
                _ => {
                    s += 1;
                    p = ep;
                    continue 'outer;
                }
            }
        };
        self.depth += 1;
        Ok(result)
    }

    /// A match of the whole pattern starting exactly at `s`.
    pub fn match_at(&mut self, s: usize) -> MatchResult {
        self.reset();
        self.do_match(s, 0)
    }
}

/// The last match on this thread: the subject's span and the captures.
#[derive(Default)]
pub struct MatchState {
    pub start: usize,
    pub end: usize,
    pub level: usize,
    pub capture: Vec<Capture>,
    pub error: String,
}

thread_local! {
    pub static STATE: RefCell<MatchState> = RefCell::new(MatchState::default());
}

fn record(m: &Matcher<'_>, start: usize, end: usize) {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        st.start = start;
        st.end = end;
        st.level = m.level;
        st.capture.clear();
        st.capture.extend_from_slice(&m.capture[..m.level]);
    });
}

fn record_error(e: String) -> i64 {
    STATE.with(|st| st.borrow_mut().error = e);
    -2
}

/// Lua's `posrelatI`: a 1-based position, negative from the end.
pub fn position(pos: i64, len: usize) -> usize {
    let len = len as i64;
    if pos > 0 {
        pos as usize
    } else if pos == 0 || pos < -len {
        1
    } else {
        (len + pos + 1) as usize
    }
}

/// `string.find`/`string.match`'s search: the first match at or after
/// byte `init`, honouring a leading `^`. The match's start, or -1 for
/// none, or -2 with the error recorded.
pub fn find(src: &[u8], pat: &[u8], init: usize) -> i64 {
    let (anchor, pat) = match pat.first() {
        Some(b'^') => (true, &pat[1..]),
        _ => (false, pat),
    };
    let mut m = Matcher::new(src, pat);
    let mut s = init;
    loop {
        match m.match_at(s) {
            Ok(Some(e)) => {
                record(&m, s, e);
                return s as i64;
            }
            Ok(None) => {}
            Err(e) => return record_error(e),
        }
        s += 1;
        if s > src.len() || anchor {
            return -1;
        }
    }
}

/// A match exactly at byte `s`, for the `gmatch`/`gsub` loops, which
/// step the position themselves: the match's end, -1 or -2.
pub fn match_here(src: &[u8], pat: &[u8], s: usize) -> i64 {
    let mut m = Matcher::new(src, pat);
    match m.match_at(s) {
        Ok(Some(e)) => {
            record(&m, s, e);
            e as i64
        }
        Ok(None) => -1,
        Err(e) => record_error(e),
    }
}

/// `%0`..`%9` and `%%` in a `gsub` replacement string, with the last
/// match's captures. An error names the misuse.
pub fn expand(src: &[u8], repl: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let mut i = 0;
    while i < repl.len() {
        let c = repl[i];
        i += 1;
        if c != ESC {
            out.push(c);
            continue;
        }
        if i >= repl.len() {
            return Err("invalid use of '%' in replacement string".into());
        }
        let d = repl[i];
        i += 1;
        match d {
            ESC => out.push(ESC),
            b'0' => {
                let (a, b) = STATE.with(|st| (st.borrow().start, st.borrow().end));
                out.extend_from_slice(&src[a..b]);
            }
            b'1'..=b'9' => match capture_text((d - b'1') as usize)? {
                CaptureValue::Bytes(a, b) => out.extend_from_slice(&src[a..b]),
                CaptureValue::Position(p) => out.extend_from_slice(p.to_string().as_bytes()),
            },
            _ => return Err("invalid use of '%' in replacement string".into()),
        }
    }
    Ok(())
}

pub enum CaptureValue {
    Bytes(usize, usize),
    Position(usize),
}

/// Capture `i` of the last match as `get_onecapture` gives it: `%0` (or
/// the only capture of a pattern without any) is the whole match.
pub fn capture_text(i: usize) -> Result<CaptureValue, String> {
    STATE.with(|st| {
        let st = st.borrow();
        if i >= st.level {
            if i == 0 {
                return Ok(CaptureValue::Bytes(st.start, st.end));
            }
            return Err(format!("invalid capture index %{}", i + 1));
        }
        let cap = st.capture[i];
        if cap.len == CAP_UNFINISHED {
            return Err("unfinished capture".into());
        }
        if cap.len == CAP_POSITION {
            return Ok(CaptureValue::Position(cap.start + 1));
        }
        Ok(CaptureValue::Bytes(cap.start, cap.start + cap.len as usize))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(s: &str, p: &str) -> Option<(usize, usize)> {
        let r = find(s.as_bytes(), p.as_bytes(), 0);
        if r < 0 {
            return None;
        }
        Some(STATE.with(|st| (st.borrow().start, st.borrow().end)))
    }

    fn caps(s: &str, p: &str) -> Vec<String> {
        let r = find(s.as_bytes(), p.as_bytes(), 0);
        assert!(r >= 0, "no match of {p:?} in {s:?}");
        let level = STATE.with(|st| st.borrow().level);
        let n = if level == 0 { 1 } else { level };
        (0..n)
            .map(|i| match capture_text(i).unwrap() {
                CaptureValue::Bytes(a, b) => s[a..b].to_string(),
                CaptureValue::Position(p) => p.to_string(),
            })
            .collect()
    }

    #[test]
    fn classes_sets_and_quantifiers() {
        assert_eq!(found("hello world", "o w"), Some((4, 7)));
        assert_eq!(found("hello", "^h.-l"), Some((0, 3)));
        assert_eq!(found("hello", "l+"), Some((2, 4)));
        assert_eq!(found("abc123", "%d+"), Some((3, 6)));
        assert_eq!(found("abc123", "[%a]+"), Some((0, 3)));
        assert_eq!(found("a-b", "[a%-]+"), Some((0, 2)));
        assert_eq!(found("x]y", "[]]"), Some((1, 2)));
        assert_eq!(found("hello", "^ello"), None);
        assert_eq!(found("hello", "lo$"), Some((3, 5)));
        assert_eq!(found("hello", "xyz"), None);
        assert_eq!(found("aaa", "a-"), Some((0, 0)));
        assert_eq!(found("", ""), Some((0, 0)));
    }

    #[test]
    fn captures_balance_and_frontier() {
        assert_eq!(caps("key=value", "(%w+)=(%w+)"), vec!["key", "value"]);
        assert_eq!(caps("hello", "()ll()"), vec!["3", "5"]);
        assert_eq!(caps("f(a(b)c)d", "%b()"), vec!["(a(b)c)"]);
        assert_eq!(caps("THE (quick) fox", "%f[%a]%a+"), vec!["THE"]);
        assert_eq!(caps("abab", "(ab)%1"), vec!["ab"]);
        assert_eq!(caps("  x", "^%s*(.-)%s*$"), vec!["x"]);
    }

    #[test]
    fn malformed_patterns_are_errors() {
        assert_eq!(find(b"x", b"%", 0), -2);
        assert_eq!(find(b"x", b"[a", 0), -2);
        // A capture left open matches; reading it is the error.
        assert_eq!(find(b"x", b"(x", 0), 0);
        assert!(capture_text(0).is_err());
        assert_eq!(find(b"x", b"%1", 0), -2);
    }

    #[test]
    fn replacement_expands_captures() {
        assert!(find(b"hello world", b"(o) (w)", 0) >= 0);
        let mut out = Vec::new();
        expand(b"hello world", b"[%2%1%%%0]", &mut out).unwrap();
        assert_eq!(out, b"[wo%o w]");
        assert!(expand(b"", b"%z", &mut out).is_err());
        assert!(expand(b"", b"%3", &mut out).is_err());
        assert!(find(b"abc", b"b", 0) >= 0);
        out.clear();
        expand(b"abc", b"<%1>", &mut out).unwrap();
        assert_eq!(out, b"<b>");
    }
}
