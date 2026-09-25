//! Lua's own reading of a chunk: the lexer and parser of the reference
//! without its code generator, so a chunk the reference refuses is
//! refused here with the reference's message and line, and the chunk
//! `full_moon` is then given is one it can take.
//!
//! A message is worded as `luaX_syntaxerror` words it: `<line>: <what>
//! near <token>`, the token as the reference shows it (`<eof>`, `'='`,
//! or the text of a name, string or numeral as read). Semantic errors
//! (labels, gotos, attributes, read-only variables) carry no token.

/// The reference's limits, as `luaconf.h` and `lparser.c` set them.
const MAX_C_CALLS: usize = 200;
const MAX_VARS: usize = 200;
const MAX_UPVALUES: usize = 255;
const MAX_REGS: usize = 255;
const FIELDS_PER_FLUSH: usize = 50;

/// Why a chunk does not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxError {
    /// The line the reference reports, `None` for a message it gives
    /// without a position (a C stack overflow).
    pub line: Option<usize>,
    /// The message after the position.
    pub message: String,
    /// Where in the source the reader stood.
    pub offset: usize,
}

impl SyntaxError {
    /// The message with its position, as `chunk:line: message`.
    pub fn with_chunk(&self, chunk: &str) -> String {
        match self.line {
            Some(line) => format!("{chunk}:{line}: {}", self.message),
            None => self.message.clone(),
        }
    }
}

/// What reading a chunk found that `full_moon` needs told.
#[derive(Debug, Default)]
pub(crate) struct Checked {
    /// Offsets of `break` statements that other statements follow in
    /// their block. Lua 5.4 allows that; `full_moon` takes a `break`
    /// only as the last statement, so each is read as `do break end`.
    pub breaks: Vec<usize>,
}

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
fn break_len(bytes: &[u8], i: usize) -> usize {
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum T {
    Char(u8),
    And,
    Break,
    Do,
    Else,
    Elseif,
    End,
    False,
    For,
    Function,
    Goto,
    If,
    In,
    Local,
    Nil,
    Not,
    Or,
    Repeat,
    Return,
    Then,
    True,
    Until,
    While,
    IDiv,
    Concat,
    Dots,
    Eq,
    Ge,
    Le,
    Ne,
    Shl,
    Shr,
    DbColon,
    Eos,
    Flt,
    Int,
    Name,
    Str,
}

const RESERVED: [(&str, T); 22] = [
    ("and", T::And),
    ("break", T::Break),
    ("do", T::Do),
    ("else", T::Else),
    ("elseif", T::Elseif),
    ("end", T::End),
    ("false", T::False),
    ("for", T::For),
    ("function", T::Function),
    ("goto", T::Goto),
    ("if", T::If),
    ("in", T::In),
    ("local", T::Local),
    ("nil", T::Nil),
    ("not", T::Not),
    ("or", T::Or),
    ("repeat", T::Repeat),
    ("return", T::Return),
    ("then", T::Then),
    ("true", T::True),
    ("until", T::Until),
    ("while", T::While),
];

/// A token as `luaX_token2str` shows it.
fn token_str(t: T) -> String {
    let fixed = match t {
        T::Char(c) if (0x20..0x7f).contains(&c) => return format!("'{}'", c as char),
        T::Char(c) => return format!("'<\\{c}>'"),
        T::Eos => return "<eof>".to_string(),
        T::Flt => return "<number>".to_string(),
        T::Int => return "<integer>".to_string(),
        T::Name => return "<name>".to_string(),
        T::Str => return "<string>".to_string(),
        T::IDiv => "//",
        T::Concat => "..",
        T::Dots => "...",
        T::Eq => "==",
        T::Ge => ">=",
        T::Le => "<=",
        T::Ne => "~=",
        T::Shl => "<<",
        T::Shr => ">>",
        T::DbColon => "::",
        word => RESERVED
            .iter()
            .find(|(_, w)| *w == word)
            .map(|(s, _)| *s)
            .unwrap_or(""),
    };
    format!("'{fixed}'")
}

#[derive(Clone, Copy)]
struct Token<'a> {
    t: T,
    /// A name's text.
    name: &'a [u8],
    /// Where the token starts.
    at: usize,
}

const NO_TOKEN: Token<'static> = Token {
    t: T::Eos,
    name: b"",
    at: 0,
};

fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_alnum(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// `lisspace`: the C locale's white space.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

fn hex_value(c: u8) -> u32 {
    (c as char).to_digit(16).unwrap_or(0)
}

/// `llex.c`: tokens on demand, with the line the reader has reached and
/// the text of the token read last, which a message quotes.
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    buf: Vec<u8>,
    tok: Token<'a>,
    ahead: Option<Token<'a>>,
}

type R<T> = Result<T, SyntaxError>;

impl<'a> Lexer<'a> {
    fn new(src: &'a [u8]) -> Self {
        Lexer {
            src,
            pos: 0,
            line: 1,
            buf: Vec::new(),
            tok: NO_TOKEN,
            ahead: None,
        }
    }

    fn cur(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.src.len() {
            self.pos += 1;
        }
    }

    fn save(&mut self, c: u8) {
        self.buf.push(c);
    }

    fn save_next(&mut self) {
        if let Some(c) = self.cur() {
            self.save(c);
        }
        self.advance();
    }

    fn cur_is_newline(&self) -> bool {
        matches!(self.cur(), Some(b'\n' | b'\r'))
    }

    fn error(&self, msg: &str, token: Option<T>) -> SyntaxError {
        let message = match token {
            Some(t) => format!("{msg} near {}", self.text_of(t)),
            None => msg.to_string(),
        };
        SyntaxError {
            line: Some(self.line),
            message,
            offset: self.pos,
        }
    }

    /// `txtToken`: a name, string or numeral as the buffer holds it.
    fn text_of(&self, t: T) -> String {
        match t {
            T::Name | T::Str | T::Flt | T::Int => {
                let end = self
                    .buf
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(self.buf.len());
                format!("'{}'", String::from_utf8_lossy(&self.buf[..end]))
            }
            t => token_str(t),
        }
    }

    fn inc_line(&mut self) {
        let old = self.cur();
        self.advance();
        if self.cur_is_newline() && self.cur() != old {
            self.advance();
        }
        self.line += 1;
    }

    fn check_next1(&mut self, c: u8) -> bool {
        if self.cur() == Some(c) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check_next2(&mut self, a: u8, b: u8) -> bool {
        if self.cur() == Some(a) || self.cur() == Some(b) {
            self.save_next();
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> R<()> {
        self.tok = match self.ahead.take() {
            Some(t) => t,
            None => self.lex()?,
        };
        Ok(())
    }

    fn lookahead(&mut self) -> R<T> {
        let t = self.lex()?;
        self.ahead = Some(t);
        Ok(t.t)
    }

    fn read_numeral(&mut self) -> R<T> {
        let first = self.cur();
        self.save_next();
        let mut expo = (b'E', b'e');
        if first == Some(b'0') && self.check_next2(b'x', b'X') {
            expo = (b'P', b'p');
        }
        loop {
            if self.check_next2(expo.0, expo.1) {
                self.check_next2(b'-', b'+');
            } else if self
                .cur()
                .is_some_and(|c| c.is_ascii_hexdigit() || c == b'.')
            {
                self.save_next();
            } else {
                break;
            }
        }
        if self.cur().is_some_and(is_alpha) {
            self.save_next();
        }
        match numeral_kind(&self.buf) {
            Some(t) => Ok(t),
            None => Err(self.error("malformed number", Some(T::Flt))),
        }
    }

    /// `[=*[` or `]=*]`: its `=` count plus two when well formed, one
    /// for a lone bracket, zero for an unfinished `[==`.
    fn skip_sep(&mut self) -> usize {
        let s = self.cur();
        let mut count = 0;
        self.save_next();
        while self.cur() == Some(b'=') {
            self.save_next();
            count += 1;
        }
        if self.cur() == s {
            count + 2
        } else if count == 0 {
            1
        } else {
            0
        }
    }

    fn read_long_string(&mut self, string: bool, sep: usize) -> R<()> {
        let line = self.line;
        self.save_next();
        if self.cur_is_newline() {
            self.inc_line();
        }
        loop {
            match self.cur() {
                None => {
                    let what = if string { "string" } else { "comment" };
                    let msg = format!("unfinished long {what} (starting at line {line})");
                    return Err(self.error(&msg, Some(T::Eos)));
                }
                Some(b']') => {
                    if self.skip_sep() == sep {
                        self.save_next();
                        return Ok(());
                    }
                }
                Some(b'\n' | b'\r') => {
                    self.save(b'\n');
                    self.inc_line();
                    if !string {
                        self.buf.clear();
                    }
                }
                Some(_) => {
                    if string {
                        self.save_next();
                    } else {
                        self.advance();
                    }
                }
            }
        }
    }

    fn esc_check(&mut self, ok: bool, msg: &str) -> R<()> {
        if ok {
            return Ok(());
        }
        if self.cur().is_some() {
            self.save_next();
        }
        Err(self.error(msg, Some(T::Str)))
    }

    fn get_hexa(&mut self) -> R<u32> {
        self.save_next();
        let c = self.cur();
        self.esc_check(
            c.is_some_and(|c| c.is_ascii_hexdigit()),
            "hexadecimal digit expected",
        )?;
        Ok(hex_value(c.unwrap_or(b'0')))
    }

    fn read_hexa_esc(&mut self) -> R<u8> {
        let r = self.get_hexa()?;
        let r = (r << 4) + self.get_hexa()?;
        self.buf.truncate(self.buf.len() - 2);
        Ok(r as u8)
    }

    fn read_utf8_esc(&mut self) -> R<u32> {
        let mut i = 4;
        self.save_next();
        self.esc_check(self.cur() == Some(b'{'), "missing '{'")?;
        let mut r = self.get_hexa()?;
        loop {
            self.save_next();
            let Some(c) = self.cur().filter(|c| c.is_ascii_hexdigit()) else {
                break;
            };
            i += 1;
            self.esc_check(r <= (0x7FFF_FFFF >> 4), "UTF-8 value too large")?;
            r = (r << 4) + hex_value(c);
        }
        self.esc_check(self.cur() == Some(b'}'), "missing '}'")?;
        self.advance();
        self.buf.truncate(self.buf.len() - i);
        Ok(r)
    }

    fn read_dec_esc(&mut self) -> R<u8> {
        let mut r: u32 = 0;
        let mut i = 0;
        while i < 3 {
            let Some(c) = self.cur().filter(u8::is_ascii_digit) else {
                break;
            };
            r = 10 * r + (c - b'0') as u32;
            self.save_next();
            i += 1;
        }
        self.esc_check(r <= 255, "decimal escape too large")?;
        self.buf.truncate(self.buf.len() - i);
        Ok(r as u8)
    }

    fn read_string(&mut self, del: u8) -> R<()> {
        self.save_next();
        while self.cur() != Some(del) {
            match self.cur() {
                None => return Err(self.error("unfinished string", Some(T::Eos))),
                Some(b'\n' | b'\r') => {
                    return Err(self.error("unfinished string", Some(T::Str)));
                }
                Some(b'\\') => {
                    self.save_next();
                    let c = match self.cur() {
                        Some(b'a') => Some((7, true)),
                        Some(b'b') => Some((8, true)),
                        Some(b'f') => Some((12, true)),
                        Some(b'n') => Some((b'\n', true)),
                        Some(b'r') => Some((b'\r', true)),
                        Some(b't') => Some((b'\t', true)),
                        Some(b'v') => Some((11, true)),
                        Some(b'x') => Some((self.read_hexa_esc()?, true)),
                        Some(b'u') => {
                            let code = self.read_utf8_esc()?;
                            for b in utf8_escape(code) {
                                self.save(b);
                            }
                            None
                        }
                        Some(b'\n' | b'\r') => {
                            self.inc_line();
                            Some((b'\n', false))
                        }
                        Some(c @ (b'\\' | b'"' | b'\'')) => Some((c, true)),
                        None => None,
                        Some(b'z') => {
                            self.buf.pop();
                            self.advance();
                            while self.cur().is_some_and(is_space) {
                                if self.cur_is_newline() {
                                    self.inc_line();
                                } else {
                                    self.advance();
                                }
                            }
                            None
                        }
                        Some(c) => {
                            self.esc_check(c.is_ascii_digit(), "invalid escape sequence")?;
                            Some((self.read_dec_esc()?, false))
                        }
                    };
                    if let Some((c, read)) = c {
                        if read {
                            self.advance();
                        }
                        self.buf.pop();
                        self.save(c);
                    }
                }
                Some(_) => self.save_next(),
            }
        }
        self.save_next();
        Ok(())
    }

    fn lex(&mut self) -> R<Token<'a>> {
        self.buf.clear();
        loop {
            let at = self.pos;
            let token = |t| Token { t, name: b"", at };
            let Some(c) = self.cur() else {
                return Ok(token(T::Eos));
            };
            match c {
                b'\n' | b'\r' => self.inc_line(),
                b' ' | b'\x0c' | b'\t' | b'\x0b' => self.advance(),
                b'-' => {
                    self.advance();
                    if self.cur() != Some(b'-') {
                        return Ok(token(T::Char(b'-')));
                    }
                    self.advance();
                    if self.cur() == Some(b'[') {
                        let sep = self.skip_sep();
                        self.buf.clear();
                        if sep >= 2 {
                            self.read_long_string(false, sep)?;
                            self.buf.clear();
                            continue;
                        }
                    }
                    while !self.cur_is_newline() && self.cur().is_some() {
                        self.advance();
                    }
                }
                b'[' => {
                    let sep = self.skip_sep();
                    if sep >= 2 {
                        self.read_long_string(true, sep)?;
                        return Ok(token(T::Str));
                    } else if sep == 0 {
                        return Err(self.error("invalid long string delimiter", Some(T::Str)));
                    }
                    return Ok(token(T::Char(b'[')));
                }
                b'=' => {
                    self.advance();
                    let t = if self.check_next1(b'=') {
                        T::Eq
                    } else {
                        T::Char(b'=')
                    };
                    return Ok(token(t));
                }
                b'<' => {
                    self.advance();
                    let t = if self.check_next1(b'=') {
                        T::Le
                    } else if self.check_next1(b'<') {
                        T::Shl
                    } else {
                        T::Char(b'<')
                    };
                    return Ok(token(t));
                }
                b'>' => {
                    self.advance();
                    let t = if self.check_next1(b'=') {
                        T::Ge
                    } else if self.check_next1(b'>') {
                        T::Shr
                    } else {
                        T::Char(b'>')
                    };
                    return Ok(token(t));
                }
                b'/' => {
                    self.advance();
                    let t = if self.check_next1(b'/') {
                        T::IDiv
                    } else {
                        T::Char(b'/')
                    };
                    return Ok(token(t));
                }
                b'~' => {
                    self.advance();
                    let t = if self.check_next1(b'=') {
                        T::Ne
                    } else {
                        T::Char(b'~')
                    };
                    return Ok(token(t));
                }
                b':' => {
                    self.advance();
                    let t = if self.check_next1(b':') {
                        T::DbColon
                    } else {
                        T::Char(b':')
                    };
                    return Ok(token(t));
                }
                b'"' | b'\'' => {
                    self.read_string(c)?;
                    return Ok(token(T::Str));
                }
                b'.' => {
                    self.save_next();
                    if self.check_next1(b'.') {
                        let t = if self.check_next1(b'.') {
                            T::Dots
                        } else {
                            T::Concat
                        };
                        return Ok(token(t));
                    }
                    if !self.cur().is_some_and(|c| c.is_ascii_digit()) {
                        return Ok(token(T::Char(b'.')));
                    }
                    return Ok(token(self.read_numeral()?));
                }
                b'0'..=b'9' => return Ok(token(self.read_numeral()?)),
                c if is_alpha(c) => {
                    while self.cur().is_some_and(is_alnum) {
                        self.save_next();
                    }
                    let name = &self.src[at..self.pos];
                    let t = RESERVED
                        .iter()
                        .find(|(w, _)| w.as_bytes() == name)
                        .map_or(T::Name, |(_, t)| *t);
                    return Ok(Token { t, name, at });
                }
                c => {
                    self.advance();
                    return Ok(token(T::Char(c)));
                }
            }
        }
    }
}

/// `luaO_utf8esc`: `x` in the extended UTF-8 the reference writes, up to
/// six bytes.
fn utf8_escape(mut x: u32) -> Vec<u8> {
    if x < 0x80 {
        return vec![x as u8];
    }
    let mut out = Vec::with_capacity(6);
    let mut mfb: u32 = 0x3f;
    loop {
        out.push(0x80 | (x & 0x3f) as u8);
        x >>= 6;
        mfb >>= 1;
        if x <= mfb {
            break;
        }
    }
    out.push(((!mfb << 1) | x) as u8);
    out.reverse();
    out
}

/// `luaO_str2num` on a numeral as read: an integer, a float, or `None`
/// when it is malformed.
fn numeral_kind(s: &[u8]) -> Option<T> {
    let hex = s.len() >= 2 && s[0] == b'0' && (s[1] == b'x' || s[1] == b'X');
    let body = if hex { &s[2..] } else { s };
    let digit = |c: &u8| {
        if hex {
            c.is_ascii_hexdigit()
        } else {
            c.is_ascii_digit()
        }
    };
    // An integer: digits alone, and for decimal, digits that fit.
    if !body.is_empty() && body.iter().all(digit) {
        if hex {
            return Some(T::Int);
        }
        let fits = std::str::from_utf8(body)
            .ok()
            .and_then(|d| d.parse::<i64>().ok())
            .is_some();
        return Some(if fits { T::Int } else { T::Flt });
    }
    // A float, as `strtod` reads it: digits with at most one point, at
    // least one digit, then an optional exponent with digits.
    let mut i = 0;
    let mut digits = 0;
    while i < body.len() && digit(&body[i]) {
        i += 1;
        digits += 1;
    }
    if i < body.len() && body[i] == b'.' {
        i += 1;
        while i < body.len() && digit(&body[i]) {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    let exponent: &[u8] = if hex { b"pP" } else { b"eE" };
    if i < body.len() && exponent.contains(&body[i]) {
        i += 1;
        if i < body.len() && (body[i] == b'+' || body[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < body.len() && body[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    (i == body.len()).then_some(T::Flt)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Regular,
    Const,
    Close,
}

struct Var<'a> {
    name: &'a [u8],
    kind: Kind,
}

#[derive(Clone, Copy)]
struct Label<'a> {
    name: &'a [u8],
    line: usize,
    nactvar: usize,
}

struct Block {
    first_label: usize,
    first_goto: usize,
    nactvar: usize,
    is_loop: bool,
}

struct Func<'a> {
    first_local: usize,
    first_label: usize,
    nactvar: usize,
    upvalues: Vec<(&'a [u8], Kind)>,
    blocks: Vec<Block>,
    vararg: bool,
    line_defined: usize,
    free_reg: usize,
}

/// What an expression is, as far as a statement needs to know: whether
/// it can be assigned, and whether it is a call.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ek {
    Void,
    Call,
    Vararg,
    Local(Kind),
    Upvalue(Kind),
    Indexed,
    Value,
}

#[derive(Clone, Copy)]
struct Exp<'a> {
    k: Ek,
    name: &'a [u8],
}

impl<'a> Exp<'a> {
    fn of(k: Ek) -> Self {
        Exp { k, name: b"" }
    }
    fn is_var(&self) -> bool {
        matches!(self.k, Ek::Local(_) | Ek::Upvalue(_) | Ek::Indexed)
    }
    fn multiple(&self) -> bool {
        matches!(self.k, Ek::Call | Ek::Vararg)
    }
}

const BREAK: &[u8] = b"break";
const ENV: &[u8] = b"_ENV";

/// `lparser.c` without code: scopes, labels, gotos, attributes and
/// the limits, over the tokens.
struct Parser<'a> {
    lx: Lexer<'a>,
    actvar: Vec<Var<'a>>,
    labels: Vec<Label<'a>>,
    gotos: Vec<Label<'a>>,
    funcs: Vec<Func<'a>>,
    level: usize,
    checked: Checked,
}

fn show(name: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(name)
}

impl<'a> Parser<'a> {
    fn t(&self) -> T {
        self.lx.tok.t
    }

    fn fs(&mut self) -> &mut Func<'a> {
        self.funcs.last_mut().expect("a function")
    }

    fn syntax_error(&self, msg: &str) -> SyntaxError {
        self.lx.error(msg, Some(self.t()))
    }

    /// `luaK_semerror`: a message without the token.
    fn sem_error(&self, msg: &str) -> SyntaxError {
        self.lx.error(msg, None)
    }

    fn error_expected(&self, t: T) -> SyntaxError {
        self.syntax_error(&format!("{} expected", token_str(t)))
    }

    fn error_limit(&self, limit: usize, what: &str) -> SyntaxError {
        let line = self.funcs.last().map_or(0, |f| f.line_defined);
        let place = if line == 0 {
            "main function".to_string()
        } else {
            format!("function at line {line}")
        };
        self.syntax_error(&format!("too many {what} (limit is {limit}) in {place}"))
    }

    fn enter_level(&mut self) -> R<()> {
        self.level += 1;
        if self.level >= MAX_C_CALLS {
            return Err(SyntaxError {
                line: None,
                message: "C stack overflow".to_string(),
                offset: self.lx.pos,
            });
        }
        Ok(())
    }

    fn leave_level(&mut self) {
        self.level -= 1;
    }

    fn next(&mut self) -> R<()> {
        self.lx.next()
    }

    fn test_next(&mut self, t: T) -> R<bool> {
        if self.t() == t {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn check(&self, t: T) -> R<()> {
        if self.t() != t {
            return Err(self.error_expected(t));
        }
        Ok(())
    }

    fn check_next(&mut self, t: T) -> R<()> {
        self.check(t)?;
        self.next()
    }

    fn check_match(&mut self, what: T, who: T, line: usize) -> R<()> {
        if self.test_next(what)? {
            return Ok(());
        }
        if line == self.lx.line {
            return Err(self.error_expected(what));
        }
        Err(self.syntax_error(&format!(
            "{} expected (to close {} at line {line})",
            token_str(what),
            token_str(who)
        )))
    }

    fn check_name(&mut self) -> R<&'a [u8]> {
        self.check(T::Name)?;
        let name = self.lx.tok.name;
        self.next()?;
        Ok(name)
    }

    fn reserve(&mut self, n: usize) -> R<()> {
        let fs = self.fs();
        fs.free_reg += n;
        if fs.free_reg >= MAX_REGS {
            return Err(self.syntax_error("function or expression needs too many registers"));
        }
        Ok(())
    }

    /// `luaK_exp2nextreg`: the expression takes the next register.
    fn put_in_next_reg(&mut self, e: &Exp) -> R<()> {
        if e.k == Ek::Void {
            return Ok(());
        }
        self.reserve(1)
    }

    fn block_follow(&self, with_until: bool) -> bool {
        match self.t() {
            T::Else | T::Elseif | T::End | T::Eos => true,
            T::Until => with_until,
            _ => false,
        }
    }

    // ── scopes ─────────────────────────────────────────────────────

    fn new_local(&mut self, name: &'a [u8]) -> R<()> {
        let first = self.funcs.last().expect("a function").first_local;
        if self.actvar.len() + 1 - first > MAX_VARS {
            return Err(self.error_limit(MAX_VARS, "local variables"));
        }
        self.actvar.push(Var {
            name,
            kind: Kind::Regular,
        });
        Ok(())
    }

    fn adjust_locals(&mut self, n: usize) {
        self.fs().nactvar += n;
    }

    fn enter_block(&mut self, is_loop: bool) {
        let (labels, gotos) = (self.labels.len(), self.gotos.len());
        let fs = self.fs();
        let nactvar = fs.nactvar;
        fs.blocks.push(Block {
            first_label: labels,
            first_goto: gotos,
            nactvar,
            is_loop,
        });
    }

    fn leave_block(&mut self) -> R<()> {
        let fs = self.funcs.last().expect("a function");
        let bl = fs.blocks.last().expect("a block");
        let (bl_nactvar, first_label, first_goto, is_loop) =
            (bl.nactvar, bl.first_label, bl.first_goto, bl.is_loop);
        let removed = fs.nactvar - bl_nactvar;
        self.actvar.truncate(self.actvar.len() - removed);
        self.fs().nactvar = bl_nactvar;
        if is_loop {
            self.create_label(BREAK, 0, false)?;
        }
        self.labels.truncate(first_label);
        let fs = self.fs();
        fs.blocks.pop();
        if !fs.blocks.is_empty() {
            for gt in &mut self.gotos[first_goto..] {
                gt.nactvar = bl_nactvar;
            }
        } else if let Some(gt) = self.gotos.get(first_goto).copied() {
            let msg = if gt.name == BREAK {
                format!("break outside loop at line {}", gt.line)
            } else {
                format!(
                    "no visible label '{}' for <goto> at line {}",
                    show(gt.name),
                    gt.line
                )
            };
            return Err(self.sem_error(&msg));
        }
        Ok(())
    }

    fn find_label(&self, name: &[u8]) -> Option<Label<'a>> {
        let first = self.funcs.last().expect("a function").first_label;
        self.labels[first..]
            .iter()
            .find(|l| l.name == name)
            .copied()
    }

    fn create_label(&mut self, name: &'a [u8], line: usize, last: bool) -> R<()> {
        let fs = self.funcs.last().expect("a function");
        let bl = fs.blocks.last().expect("a block");
        let nactvar = if last { bl.nactvar } else { fs.nactvar };
        let (first_local, first_goto) = (fs.first_local, bl.first_goto);
        self.labels.push(Label {
            name,
            line,
            nactvar,
        });
        let mut i = first_goto;
        while i < self.gotos.len() {
            let gt = self.gotos[i];
            if gt.name != name {
                i += 1;
                continue;
            }
            if gt.nactvar < nactvar {
                let var = show(self.actvar[first_local + gt.nactvar].name);
                let msg = format!(
                    "<goto {}> at line {} jumps into the scope of local '{var}'",
                    show(gt.name),
                    gt.line
                );
                return Err(self.sem_error(&msg));
            }
            self.gotos.remove(i);
        }
        Ok(())
    }

    fn open_func(&mut self, line_defined: usize) {
        self.funcs.push(Func {
            first_local: self.actvar.len(),
            first_label: self.labels.len(),
            nactvar: 0,
            upvalues: Vec::new(),
            blocks: Vec::new(),
            vararg: false,
            line_defined,
            free_reg: 0,
        });
        self.enter_block(false);
    }

    fn close_func(&mut self) -> R<()> {
        self.leave_block()?;
        self.funcs.pop();
        Ok(())
    }

    /// The variable `name` is at level `f`: a local, an upvalue (made in
    /// every function between), or `None` for a global.
    fn resolve(&mut self, f: usize, name: &'a [u8]) -> R<Option<Ek>> {
        let fs = &self.funcs[f];
        for i in (0..fs.nactvar).rev() {
            let var = &self.actvar[fs.first_local + i];
            if var.name == name {
                return Ok(Some(Ek::Local(var.kind)));
            }
        }
        if let Some((_, kind)) = fs.upvalues.iter().find(|(n, _)| *n == name) {
            return Ok(Some(Ek::Upvalue(*kind)));
        }
        if f == 0 {
            return Ok(None);
        }
        let kind = match self.resolve(f - 1, name)? {
            Some(Ek::Local(kind) | Ek::Upvalue(kind)) => kind,
            _ => return Ok(None),
        };
        if self.funcs[f].upvalues.len() + 1 > MAX_UPVALUES {
            let line = self.funcs[f].line_defined;
            let place = if line == 0 {
                "main function".to_string()
            } else {
                format!("function at line {line}")
            };
            return Err(self.syntax_error(&format!(
                "too many upvalues (limit is {MAX_UPVALUES}) in {place}"
            )));
        }
        self.funcs[f].upvalues.push((name, kind));
        Ok(Some(Ek::Upvalue(kind)))
    }

    fn single_var(&mut self) -> R<Exp<'a>> {
        let name = self.check_name()?;
        let f = self.funcs.len() - 1;
        match self.resolve(f, name)? {
            Some(k) => Ok(Exp { k, name }),
            None => {
                self.resolve(f, ENV)?;
                Ok(Exp {
                    k: Ek::Indexed,
                    name,
                })
            }
        }
    }

    fn check_readonly(&self, e: &Exp) -> R<()> {
        match e.k {
            Ek::Local(k) | Ek::Upvalue(k) if k != Kind::Regular => Err(self.sem_error(&format!(
                "attempt to assign to const variable '{}'",
                show(e.name)
            ))),
            _ => Ok(()),
        }
    }

    // ── statements ─────────────────────────────────────────────────

    fn main(&mut self) -> R<()> {
        self.open_func(0);
        let fs = self.fs();
        fs.vararg = true;
        fs.upvalues.push((ENV, Kind::Regular));
        self.next()?;
        self.statlist()?;
        self.check(T::Eos)?;
        self.close_func()
    }

    fn statlist(&mut self) -> R<()> {
        while !self.block_follow(true) {
            if self.t() == T::Return {
                return self.statement();
            }
            self.statement()?;
        }
        Ok(())
    }

    fn block(&mut self) -> R<()> {
        self.enter_block(false);
        self.statlist()?;
        self.leave_block()
    }

    fn statement(&mut self) -> R<()> {
        let line = self.lx.line;
        self.enter_level()?;
        match self.t() {
            T::Char(b';') => self.next()?,
            T::If => self.if_stat(line)?,
            T::While => self.while_stat(line)?,
            T::Do => {
                self.next()?;
                self.block()?;
                self.check_match(T::End, T::Do, line)?;
            }
            T::For => self.for_stat(line)?,
            T::Repeat => self.repeat_stat(line)?,
            T::Function => self.func_stat(line)?,
            T::Local => {
                self.next()?;
                if self.test_next(T::Function)? {
                    self.local_func()?;
                } else {
                    self.local_stat()?;
                }
            }
            T::DbColon => {
                self.next()?;
                let name = self.check_name()?;
                self.label_stat(name, line)?;
            }
            T::Return => {
                self.next()?;
                self.ret_stat()?;
            }
            T::Break => {
                let at = self.lx.tok.at;
                let line = self.lx.line;
                self.next()?;
                let nactvar = self.fs().nactvar;
                self.gotos.push(Label {
                    name: BREAK,
                    line,
                    nactvar,
                });
                if !self.block_follow(true) {
                    self.checked.breaks.push(at);
                }
            }
            T::Goto => {
                self.next()?;
                self.goto_stat()?;
            }
            _ => self.expr_stat()?,
        }
        let fs = self.fs();
        fs.free_reg = fs.nactvar;
        self.leave_level();
        Ok(())
    }

    fn if_stat(&mut self, line: usize) -> R<()> {
        self.test_then_block()?;
        while self.t() == T::Elseif {
            self.test_then_block()?;
        }
        if self.test_next(T::Else)? {
            self.block()?;
        }
        self.check_match(T::End, T::If, line)
    }

    fn test_then_block(&mut self) -> R<()> {
        self.next()?;
        self.expr()?;
        self.check_next(T::Then)?;
        self.enter_block(false);
        self.statlist()?;
        self.leave_block()
    }

    fn while_stat(&mut self, line: usize) -> R<()> {
        self.next()?;
        self.expr()?;
        self.enter_block(true);
        self.check_next(T::Do)?;
        self.block()?;
        self.check_match(T::End, T::While, line)?;
        self.leave_block()
    }

    fn repeat_stat(&mut self, line: usize) -> R<()> {
        self.enter_block(true);
        self.enter_block(false);
        self.next()?;
        self.statlist()?;
        self.check_match(T::Until, T::Repeat, line)?;
        self.expr()?;
        self.leave_block()?;
        self.leave_block()
    }

    fn exp1(&mut self) -> R<()> {
        let e = self.expr()?;
        self.put_in_next_reg(&e)
    }

    fn for_stat(&mut self, line: usize) -> R<()> {
        self.enter_block(true);
        self.next()?;
        let name = self.check_name()?;
        match self.t() {
            T::Char(b'=') => {
                for _ in 0..3 {
                    self.new_local(b"(for state)")?;
                }
                self.new_local(name)?;
                self.check_next(T::Char(b'='))?;
                self.exp1()?;
                self.check_next(T::Char(b','))?;
                self.exp1()?;
                if self.test_next(T::Char(b','))? {
                    self.exp1()?;
                } else {
                    self.reserve(1)?;
                }
                self.adjust_locals(3);
                self.for_body(1)?;
            }
            T::Char(b',') | T::In => {
                for _ in 0..4 {
                    self.new_local(b"(for state)")?;
                }
                self.new_local(name)?;
                let mut nvars = 5;
                while self.test_next(T::Char(b','))? {
                    let name = self.check_name()?;
                    self.new_local(name)?;
                    nvars += 1;
                }
                self.check_next(T::In)?;
                self.explist()?;
                self.adjust_locals(4);
                self.for_body(nvars - 4)?;
            }
            _ => return Err(self.syntax_error("'=' or 'in' expected")),
        }
        self.check_match(T::End, T::For, line)?;
        self.leave_block()
    }

    fn for_body(&mut self, nvars: usize) -> R<()> {
        self.check_next(T::Do)?;
        self.enter_block(false);
        self.adjust_locals(nvars);
        self.block()?;
        self.leave_block()
    }

    fn func_stat(&mut self, line: usize) -> R<()> {
        self.next()?;
        let mut v = self.single_var()?;
        let mut method = false;
        while self.t() == T::Char(b'.') {
            self.field_sel()?;
            v = Exp::of(Ek::Indexed);
        }
        if self.t() == T::Char(b':') {
            method = true;
            self.field_sel()?;
            v = Exp::of(Ek::Indexed);
        }
        self.body(method, line)?;
        self.check_readonly(&v)
    }

    fn local_func(&mut self) -> R<()> {
        let name = self.check_name()?;
        self.new_local(name)?;
        self.adjust_locals(1);
        let line = self.lx.line;
        self.body(false, line)
    }

    fn local_stat(&mut self) -> R<()> {
        let mut to_close = false;
        let mut nvars = 0;
        loop {
            let name = self.check_name()?;
            self.new_local(name)?;
            let kind = self.local_attribute()?;
            self.actvar.last_mut().expect("the local").kind = kind;
            if kind == Kind::Close {
                if to_close {
                    return Err(self.sem_error("multiple to-be-closed variables in local list"));
                }
                to_close = true;
            }
            nvars += 1;
            if !self.test_next(T::Char(b','))? {
                break;
            }
        }
        if self.test_next(T::Char(b'='))? {
            self.explist()?;
        }
        self.adjust_locals(nvars);
        Ok(())
    }

    fn local_attribute(&mut self) -> R<Kind> {
        if !self.test_next(T::Char(b'<'))? {
            return Ok(Kind::Regular);
        }
        let attr = self.check_name()?;
        self.check_next(T::Char(b'>'))?;
        match attr {
            b"const" => Ok(Kind::Const),
            b"close" => Ok(Kind::Close),
            other => Err(self.sem_error(&format!("unknown attribute '{}'", show(other)))),
        }
    }

    fn label_stat(&mut self, name: &'a [u8], line: usize) -> R<()> {
        self.check_next(T::DbColon)?;
        while matches!(self.t(), T::Char(b';') | T::DbColon) {
            self.statement()?;
        }
        if let Some(lb) = self.find_label(name) {
            return Err(self.sem_error(&format!(
                "label '{}' already defined on line {}",
                show(name),
                lb.line
            )));
        }
        let last = self.block_follow(false);
        self.create_label(name, line, last)
    }

    fn goto_stat(&mut self) -> R<()> {
        let line = self.lx.line;
        let name = self.check_name()?;
        if self.find_label(name).is_none() {
            let nactvar = self.fs().nactvar;
            self.gotos.push(Label {
                name,
                line,
                nactvar,
            });
        }
        Ok(())
    }

    fn ret_stat(&mut self) -> R<()> {
        if !self.block_follow(true) && self.t() != T::Char(b';') {
            self.explist()?;
        }
        self.test_next(T::Char(b';'))?;
        Ok(())
    }

    fn expr_stat(&mut self) -> R<()> {
        let v = self.suffixed_exp()?;
        if matches!(self.t(), T::Char(b'=' | b',')) {
            self.rest_assign(v)
        } else if v.k != Ek::Call {
            Err(self.syntax_error("syntax error"))
        } else {
            Ok(())
        }
    }

    fn rest_assign(&mut self, lh: Exp<'a>) -> R<()> {
        if !lh.is_var() {
            return Err(self.syntax_error("syntax error"));
        }
        self.check_readonly(&lh)?;
        if self.test_next(T::Char(b','))? {
            let nv = self.suffixed_exp()?;
            self.enter_level()?;
            self.rest_assign(nv)?;
            self.leave_level();
        } else {
            self.check_next(T::Char(b'='))?;
            self.explist()?;
        }
        Ok(())
    }

    // ── expressions ────────────────────────────────────────────────

    fn explist(&mut self) -> R<Exp<'a>> {
        let mut e = self.expr()?;
        while self.test_next(T::Char(b','))? {
            self.put_in_next_reg(&e)?;
            e = self.expr()?;
        }
        Ok(e)
    }

    fn field_sel(&mut self) -> R<()> {
        self.next()?;
        self.check_name()?;
        Ok(())
    }

    fn index(&mut self) -> R<()> {
        self.next()?;
        self.expr()?;
        self.check_next(T::Char(b']'))
    }

    fn body(&mut self, method: bool, line: usize) -> R<()> {
        self.open_func(line);
        self.check_next(T::Char(b'('))?;
        if method {
            self.new_local(b"self")?;
            self.adjust_locals(1);
        }
        self.parlist()?;
        self.check_next(T::Char(b')'))?;
        self.statlist()?;
        self.check_match(T::End, T::Function, line)?;
        self.close_func()
    }

    fn parlist(&mut self) -> R<()> {
        let mut nparams = 0;
        let mut vararg = false;
        if self.t() != T::Char(b')') {
            loop {
                match self.t() {
                    T::Name => {
                        let name = self.check_name()?;
                        self.new_local(name)?;
                        nparams += 1;
                    }
                    T::Dots => {
                        self.next()?;
                        vararg = true;
                    }
                    _ => return Err(self.syntax_error("<name> or '...' expected")),
                }
                if vararg || !self.test_next(T::Char(b','))? {
                    break;
                }
            }
        }
        self.adjust_locals(nparams);
        let fs = self.fs();
        fs.vararg = vararg;
        let n = fs.nactvar;
        self.reserve(n)
    }

    fn constructor(&mut self) -> R<()> {
        let line = self.lx.line;
        self.reserve(1)?;
        let table_reg = self.fs().free_reg;
        self.check_next(T::Char(b'{'))?;
        let mut pending = Exp::of(Ek::Void);
        let mut to_store = 0;
        loop {
            if self.t() == T::Char(b'}') {
                break;
            }
            if pending.k != Ek::Void {
                self.put_in_next_reg(&pending)?;
                pending = Exp::of(Ek::Void);
                if to_store == FIELDS_PER_FLUSH {
                    self.fs().free_reg = table_reg;
                    to_store = 0;
                }
            }
            let list = match self.t() {
                T::Name => self.lx.lookahead()? != T::Char(b'='),
                T::Char(b'[') => false,
                _ => true,
            };
            if list {
                pending = self.expr()?;
                to_store += 1;
            } else {
                let reg = self.fs().free_reg;
                if self.t() == T::Name {
                    self.check_name()?;
                } else {
                    self.index()?;
                }
                self.check_next(T::Char(b'='))?;
                self.expr()?;
                self.fs().free_reg = reg;
            }
            if !(self.test_next(T::Char(b','))? || self.test_next(T::Char(b';'))?) {
                break;
            }
        }
        self.check_match(T::Char(b'}'), T::Char(b'{'), line)
    }

    /// The arguments of a call whose function is in register `base`.
    fn func_args(&mut self, line: usize, base: usize) -> R<()> {
        match self.t() {
            T::Char(b'(') => {
                self.next()?;
                if self.t() != T::Char(b')') {
                    let e = self.explist()?;
                    if !e.multiple() {
                        self.put_in_next_reg(&e)?;
                    }
                }
                self.check_match(T::Char(b')'), T::Char(b'('), line)?;
            }
            T::Char(b'{') => self.constructor()?,
            T::Str => self.next()?,
            _ => return Err(self.syntax_error("function arguments expected")),
        }
        self.fs().free_reg = base + 1;
        Ok(())
    }

    fn primary_exp(&mut self) -> R<Exp<'a>> {
        match self.t() {
            T::Char(b'(') => {
                let line = self.lx.line;
                self.next()?;
                self.expr()?;
                self.check_match(T::Char(b')'), T::Char(b'('), line)?;
                Ok(Exp::of(Ek::Value))
            }
            T::Name => self.single_var(),
            _ => Err(self.syntax_error("unexpected symbol")),
        }
    }

    fn suffixed_exp(&mut self) -> R<Exp<'a>> {
        let mut v = self.primary_exp()?;
        loop {
            match self.t() {
                T::Char(b'.') => {
                    self.field_sel()?;
                    v = Exp::of(Ek::Indexed);
                }
                T::Char(b'[') => {
                    self.index()?;
                    v = Exp::of(Ek::Indexed);
                }
                T::Char(b':') => {
                    let line = self.lx.line;
                    self.next()?;
                    self.check_name()?;
                    let base = self.fs().free_reg;
                    self.reserve(2)?;
                    self.func_args(line, base)?;
                    v = Exp::of(Ek::Call);
                }
                T::Char(b'(' | b'{') | T::Str => {
                    let line = self.lx.line;
                    let base = self.fs().free_reg;
                    self.put_in_next_reg(&v)?;
                    self.func_args(line, base)?;
                    v = Exp::of(Ek::Call);
                }
                _ => return Ok(v),
            }
        }
    }

    fn simple_exp(&mut self) -> R<Exp<'a>> {
        let e = match self.t() {
            T::Flt | T::Int | T::Str | T::Nil | T::True | T::False => Exp::of(Ek::Value),
            T::Dots => {
                if !self.funcs.last().expect("a function").vararg {
                    return Err(self.syntax_error("cannot use '...' outside a vararg function"));
                }
                Exp::of(Ek::Vararg)
            }
            T::Char(b'{') => {
                self.constructor()?;
                return Ok(Exp::of(Ek::Value));
            }
            T::Function => {
                self.next()?;
                let line = self.lx.line;
                self.body(false, line)?;
                return Ok(Exp::of(Ek::Value));
            }
            _ => return self.suffixed_exp(),
        };
        self.next()?;
        Ok(e)
    }

    fn expr(&mut self) -> R<Exp<'a>> {
        self.subexpr(0).map(|(e, _)| e)
    }

    /// `subexpr`: operators binding tighter than `limit`, and the first
    /// operator it left.
    fn subexpr(&mut self, limit: u8) -> R<(Exp<'a>, Option<(u8, u8)>)> {
        self.enter_level()?;
        let mut e = if matches!(self.t(), T::Not | T::Char(b'-' | b'~' | b'#')) {
            self.next()?;
            self.subexpr(UNARY_PRIORITY)?;
            Exp::of(Ek::Value)
        } else {
            self.simple_exp()?
        };
        let mut op = binary_priority(self.t());
        while let Some((left, right)) = op {
            if left <= limit {
                break;
            }
            self.next()?;
            let (_, next) = self.subexpr(right)?;
            e = Exp::of(Ek::Value);
            op = next;
        }
        self.leave_level();
        Ok((e, op))
    }
}

const UNARY_PRIORITY: u8 = 12;

/// A binary operator's left and right priority, from `lparser.c`.
fn binary_priority(t: T) -> Option<(u8, u8)> {
    Some(match t {
        T::Char(b'+' | b'-') => (10, 10),
        T::Char(b'*' | b'%' | b'/') | T::IDiv => (11, 11),
        T::Char(b'^') => (14, 13),
        T::Char(b'&') => (6, 6),
        T::Char(b'|') => (4, 4),
        T::Char(b'~') => (5, 5),
        T::Shl | T::Shr => (7, 7),
        T::Concat => (9, 8),
        T::Eq | T::Char(b'<' | b'>') | T::Le | T::Ne | T::Ge => (3, 3),
        T::And => (2, 2),
        T::Or => (1, 1),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(src: &str) -> String {
        check(src.as_bytes(), 2)
            .map(|_| "ok".to_string())
            .unwrap_or_else(|e| e.with_chunk("c"))
    }

    #[test]
    fn messages_match_the_reference() {
        assert_eq!(err("x = "), "c:1: unexpected symbol near <eof>");
        assert_eq!(
            err("local a = {4\n\n"),
            "c:3: '}' expected (to close '{' at line 1) near <eof>"
        );
        assert_eq!(err("syntax error"), "c:1: syntax error near 'error'");
        assert_eq!(err("1.000"), "c:1: unexpected symbol near '1.000'");
        assert_eq!(err("x = 'abc"), "c:1: unfinished string near <eof>");
        assert_eq!(err("x = 3x"), "c:1: malformed number near '3x'");
        assert_eq!(err("return;;"), "c:1: <eof> expected near ';'");
        assert_eq!(err("\u{1}a = 1"), "c:1: unexpected symbol near '<\\1>'");
        assert_eq!(
            err("goto f; local x; ::f:: print(x)"),
            "c:1: <goto f> at line 1 jumps into the scope of local 'x'"
        );
        assert_eq!(err("break"), "c:1: break outside loop at line 1");
        assert_eq!(
            err("local x <const> = 1; x = 2"),
            "c:1: attempt to assign to const variable 'x'"
        );
        assert_eq!(err("f() = 1"), "c:1: syntax error near '='");
    }

    #[test]
    fn break_may_be_followed() {
        let checked = check(b"while x do break; x = 1 end", 2).expect("parses");
        assert_eq!(checked.breaks, vec![11]);
        let checked = check(b"while x do break end", 2).expect("parses");
        assert!(checked.breaks.is_empty());
    }
}
