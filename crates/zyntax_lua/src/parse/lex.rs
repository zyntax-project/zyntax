//! Lex support.

use super::*;

/// A token as `luaX_token2str` shows it.
pub(super) fn token_str(t: T) -> String {
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

pub(super) fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

pub(super) fn is_alnum(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// `lisspace`: the C locale's white space.
pub(super) fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

pub(super) fn hex_value(c: u8) -> u32 {
    (c as char).to_digit(16).unwrap_or(0)
}

impl<'a> Lexer<'a> {
    pub(super) fn new(src: &'a [u8]) -> Self {
        Lexer {
            src,
            pos: 0,
            line: 1,
            buf: Vec::new(),
            tok: NO_TOKEN,
            ahead: None,
        }
    }

    pub(super) fn cur(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    pub(super) fn advance(&mut self) {
        if self.pos < self.src.len() {
            self.pos += 1;
        }
    }

    pub(super) fn save(&mut self, c: u8) {
        self.buf.push(c);
    }

    pub(super) fn save_next(&mut self) {
        if let Some(c) = self.cur() {
            self.save(c);
        }
        self.advance();
    }

    pub(super) fn cur_is_newline(&self) -> bool {
        matches!(self.cur(), Some(b'\n' | b'\r'))
    }

    pub(super) fn error(&self, msg: &str, token: Option<T>) -> SyntaxError {
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
    pub(super) fn text_of(&self, t: T) -> String {
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

    pub(super) fn inc_line(&mut self) {
        let old = self.cur();
        self.advance();
        if self.cur_is_newline() && self.cur() != old {
            self.advance();
        }
        self.line += 1;
    }

    pub(super) fn check_next1(&mut self, c: u8) -> bool {
        if self.cur() == Some(c) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub(super) fn check_next2(&mut self, a: u8, b: u8) -> bool {
        if self.cur() == Some(a) || self.cur() == Some(b) {
            self.save_next();
            true
        } else {
            false
        }
    }

    pub(super) fn next(&mut self) -> R<()> {
        self.tok = match self.ahead.take() {
            Some(t) => t,
            None => self.lex()?,
        };
        Ok(())
    }

    pub(super) fn lookahead(&mut self) -> R<T> {
        let t = self.lex()?;
        self.ahead = Some(t);
        Ok(t.t)
    }

    pub(super) fn read_numeral(&mut self) -> R<T> {
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
    pub(super) fn skip_sep(&mut self) -> usize {
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

    pub(super) fn read_long_string(&mut self, string: bool, sep: usize) -> R<()> {
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

    pub(super) fn esc_check(&mut self, ok: bool, msg: &str) -> R<()> {
        if ok {
            return Ok(());
        }
        if self.cur().is_some() {
            self.save_next();
        }
        Err(self.error(msg, Some(T::Str)))
    }

    pub(super) fn get_hexa(&mut self) -> R<u32> {
        self.save_next();
        let c = self.cur();
        self.esc_check(
            c.is_some_and(|c| c.is_ascii_hexdigit()),
            "hexadecimal digit expected",
        )?;
        Ok(hex_value(c.unwrap_or(b'0')))
    }

    pub(super) fn read_hexa_esc(&mut self) -> R<u8> {
        let r = self.get_hexa()?;
        let r = (r << 4) + self.get_hexa()?;
        self.buf.truncate(self.buf.len() - 2);
        Ok(r as u8)
    }

    pub(super) fn read_utf8_esc(&mut self) -> R<u32> {
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

    pub(super) fn read_dec_esc(&mut self) -> R<u8> {
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

    pub(super) fn read_string(&mut self, del: u8) -> R<()> {
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

    pub(super) fn lex(&mut self) -> R<Token<'a>> {
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
pub(super) fn utf8_escape(mut x: u32) -> Vec<u8> {
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
pub(super) fn numeral_kind(s: &[u8]) -> Option<T> {
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
