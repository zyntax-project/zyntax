//! Source support.

use super::*;

/// Read `source` as the reference reads it. `level` is the C call depth
/// the reader starts at, which the reference's nesting limit counts.
pub(crate) fn check(source: &[u8], level: usize) -> Result<Checked, SyntaxError> {
    let mut p = Parser {
        lx: Lexer::new(source),
        actvar: Vec::new(),
        labels: Vec::new(),
        gotos: Vec::new(),
        funcs: Vec::new(),
        level,
        checked: Checked::default(),
    };
    p.main()?;
    Ok(p.checked)
}

/// The length of the line break starting at `bytes[i]`: `\n`, `\r`,
/// `\n\r` and `\r\n` are each one break, as the reference's lexer
/// counts them; 0 when no break starts there.
pub(super) fn break_len(bytes: &[u8], i: usize) -> usize {
    match bytes.get(i) {
        Some(&c @ (b'\n' | b'\r')) => match bytes.get(i + 1) {
            Some(&d @ (b'\n' | b'\r')) if d != c => 2,
            _ => 1,
        },
        _ => 0,
    }
}

/// `text` with every line break a single `\n`, which is what a long
/// string holds for one and what a line count sees; `None` when it
/// already is (it has no `\r`).
pub(crate) fn with_newline_breaks(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let first = text.find('\r')?;
    // A `\n` just before the first `\r` begins its break.
    let mut i = if first > 0 && bytes[first - 1] == b'\n' {
        first - 1
    } else {
        first
    };
    let mut out = Vec::with_capacity(bytes.len());
    out.extend_from_slice(&bytes[..i]);
    while i < bytes.len() {
        match break_len(bytes, i) {
            0 => {
                out.push(bytes[i]);
                i += 1;
            }
            n => {
                out.push(b'\n');
                i += n;
            }
        }
    }
    Some(String::from_utf8(out).expect("line breaks replaced by a byte of ASCII"))
}

/// The offset in `original` of what is at `at` in its
/// [`with_newline_breaks`] text.
pub(crate) fn original_offset(original: &str, at: usize) -> usize {
    let bytes = original.as_bytes();
    let mut i = 0;
    for _ in 0..at {
        if i >= bytes.len() {
            break;
        }
        i += break_len(bytes, i).max(1);
    }
    i
}

/// The line `at` is on in `source`, counting breaks as the reference
/// does.
pub(crate) fn line_at(source: &str, at: usize) -> usize {
    let bytes = &source.as_bytes()[..at.min(source.len())];
    let mut line = 1;
    let mut i = 0;
    while let Some(k) = bytes[i..].iter().position(|&b| b == b'\n' || b == b'\r') {
        line += 1;
        i += k + break_len(bytes, i + k);
    }
    line
}

/// `source` with every `break` in `checked.breaks` made a block of its
/// own, so a statement may follow it. Lines do not move.
pub(crate) fn with_breaks_closed(source: &[u8], checked: &Checked) -> Option<Vec<u8>> {
    if checked.breaks.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(source.len() + 8 * checked.breaks.len());
    let mut from = 0;
    for &at in &checked.breaks {
        out.extend_from_slice(&source[from..at]);
        out.extend_from_slice(b"do break end");
        from = at + b"break".len();
    }
    out.extend_from_slice(&source[from..]);
    Some(out)
}
