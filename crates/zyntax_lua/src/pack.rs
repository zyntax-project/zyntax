//! `string.pack`, `string.unpack` and `string.packsize`: the format
//! language of the reference's lstrlib, over bytes. Sizes are those of
//! a 64-bit C platform, little-endian by default, and the reference's
//! messages are given word for word.

use crate::host::Numeral;

/// Lua's integer size, in bytes.
const SZINT: usize = 8;
/// The largest integer an `i`/`I` option may name.
const MAXINTSIZE: usize = 16;
/// The natural alignment of the largest C scalar, what `!` alone
/// asks for; until then nothing is aligned.
const MAXALIGN: usize = 8;
/// What alignment pads with.
const PADBYTE: u8 = 0;
/// The largest size a format may describe, as the reference bounds
/// it by a C int.
const MAXSIZE: i64 = i32::MAX as i64;

/// One argument as pack reads it: an integer or a float, kept as
/// given so a conversion can fail with the right words; a string as
/// its bytes; anything else by its type name.
pub enum PackArg<'a> {
    Int(i64),
    Float(f64),
    Str(&'a [u8]),
    Other(&'static str),
}

/// A value `unpack` produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Unpacked {
    Int(i64),
    Float(f64),
    Str(Vec<u8>),
}

enum Option_ {
    Int,
    Uint,
    Float,
    Number,
    Double,
    Char,
    String,
    Zstr,
    Padding,
    Paddalign,
    Nop,
}

struct Header {
    little: bool,
    maxalign: usize,
}

struct Format<'a> {
    fmt: &'a [u8],
    at: usize,
}

impl Format<'_> {
    fn peek(&self) -> Option<u8> {
        self.fmt.get(self.at).copied()
    }

    /// A run of digits, or `default` without any.
    fn number(&mut self, default: i64) -> i64 {
        if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return default;
        }
        let mut a: i64 = 0;
        while let Some(c) = self.peek()
            && c.is_ascii_digit()
            && a <= (MAXSIZE - 9) / 10
        {
            a = a * 10 + i64::from(c - b'0');
            self.at += 1;
        }
        a
    }

    fn number_limit(&mut self, default: usize) -> Result<usize, String> {
        let size = self.number(default as i64);
        if size > MAXINTSIZE as i64 || size <= 0 {
            return Err(format!(
                "integral size ({size}) out of limits [1,{MAXINTSIZE}]"
            ));
        }
        Ok(size as usize)
    }

    /// The next option and its size.
    fn option(&mut self, h: &mut Header) -> Result<(Option_, usize), String> {
        let opt = self.fmt[self.at];
        self.at += 1;
        Ok(match opt {
            b'b' => (Option_::Int, 1),
            b'B' => (Option_::Uint, 1),
            b'h' => (Option_::Int, 2),
            b'H' => (Option_::Uint, 2),
            b'l' => (Option_::Int, 8),
            b'L' => (Option_::Uint, 8),
            b'j' => (Option_::Int, 8),
            b'J' => (Option_::Uint, 8),
            b'T' => (Option_::Uint, 8),
            b'f' => (Option_::Float, 4),
            b'n' => (Option_::Number, 8),
            b'd' => (Option_::Double, 8),
            b'i' => (Option_::Int, self.number_limit(4)?),
            b'I' => (Option_::Uint, self.number_limit(4)?),
            b's' => (Option_::String, self.number_limit(8)?),
            b'c' => {
                let size = self.number(-1);
                if size == -1 {
                    return Err("missing size for format option 'c'".to_string());
                }
                (Option_::Char, size as usize)
            }
            b'z' => (Option_::Zstr, 0),
            b'x' => (Option_::Padding, 1),
            b'X' => (Option_::Paddalign, 0),
            b' ' => (Option_::Nop, 0),
            b'<' => {
                h.little = true;
                (Option_::Nop, 0)
            }
            b'>' => {
                h.little = false;
                (Option_::Nop, 0)
            }
            b'=' => {
                h.little = true;
                (Option_::Nop, 0)
            }
            b'!' => {
                h.maxalign = self.number_limit(MAXALIGN)?;
                (Option_::Nop, 0)
            }
            other => {
                return Err(format!("invalid format option '{}'", other as char));
            }
        })
    }

    /// The next option, its size, and the padding that aligns it after
    /// `total` bytes.
    fn details(&mut self, h: &mut Header, total: usize) -> Result<(Option_, usize, usize), String> {
        let (opt, size) = self.option(h)?;
        let mut align = size;
        if matches!(opt, Option_::Paddalign) {
            if self.peek().is_none() {
                return Err(arg_error(1, "invalid next option for option 'X'"));
            }
            let (next, next_size) = self.option(h)?;
            align = next_size;
            if matches!(next, Option_::Char) || align == 0 {
                return Err(arg_error(1, "invalid next option for option 'X'"));
            }
        }
        let to_align = if align <= 1 || matches!(opt, Option_::Char) {
            0
        } else {
            if align > h.maxalign {
                align = h.maxalign;
            }
            if align & (align - 1) != 0 {
                return Err(arg_error(1, "format asks for alignment not power of 2"));
            }
            (align - (total & (align - 1))) & (align - 1)
        };
        Ok((opt, size, to_align))
    }
}

fn arg_error(n: usize, message: &str) -> String {
    format!("bad argument #{n} to 'pack' ({message})")
}

fn header() -> Header {
    Header {
        little: true,
        maxalign: 1,
    }
}

fn pack_int(out: &mut Vec<u8>, n: u64, little: bool, size: usize, negative: bool) {
    let start = out.len();
    out.resize(start + size, 0);
    let buf = &mut out[start..];
    let mut n = n;
    buf[if little { 0 } else { size - 1 }] = (n & 0xff) as u8;
    for i in 1..size {
        n >>= 8;
        buf[if little { i } else { size - 1 - i }] = (n & 0xff) as u8;
    }
    if negative && size > SZINT {
        for i in SZINT..size {
            buf[if little { i } else { size - 1 - i }] = 0xff;
        }
    }
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8], little: bool) {
    if little {
        out.extend_from_slice(bytes);
    } else {
        out.extend(bytes.iter().rev());
    }
}

fn int_of(a: &PackArg<'_>, n: usize) -> Result<i64, String> {
    match a {
        PackArg::Int(v) => Ok(*v),
        PackArg::Float(v) => float_int(*v, n),
        PackArg::Str(s) => match crate::host::parse_numeral(&String::from_utf8_lossy(s)) {
            Numeral::Int(v) => Ok(v),
            Numeral::Float(v) => float_int(v, n),
            Numeral::None => Err(arg_error(n, "number expected, got string")),
        },
        PackArg::Other(t) => Err(arg_error(n, &format!("number expected, got {t}"))),
    }
}

fn float_int(v: f64, n: usize) -> Result<i64, String> {
    if v.fract() == 0.0 && (-9223372036854775808.0..9223372036854775808.0).contains(&v) {
        Ok(v as i64)
    } else {
        Err(arg_error(n, "number has no integer representation"))
    }
}

fn float_of(a: &PackArg<'_>, n: usize) -> Result<f64, String> {
    match a {
        PackArg::Int(v) => Ok(*v as f64),
        PackArg::Float(v) => Ok(*v),
        PackArg::Str(s) => match crate::host::parse_numeral(&String::from_utf8_lossy(s)) {
            Numeral::Int(v) => Ok(v as f64),
            Numeral::Float(v) => Ok(v),
            Numeral::None => Err(arg_error(n, "number expected, got string")),
        },
        PackArg::Other(t) => Err(arg_error(n, &format!("number expected, got {t}"))),
    }
}

fn bytes_of<'a>(a: &'a PackArg<'a>, n: usize) -> Result<std::borrow::Cow<'a, [u8]>, String> {
    match a {
        PackArg::Str(s) => Ok(std::borrow::Cow::Borrowed(s)),
        PackArg::Int(v) => Ok(std::borrow::Cow::Owned(v.to_string().into_bytes())),
        PackArg::Float(v) => Ok(std::borrow::Cow::Owned(
            crate::host::float_text(*v).into_bytes(),
        )),
        PackArg::Other(t) => Err(arg_error(n, &format!("string expected, got {t}"))),
    }
}

/// `string.pack(fmt, ...)`.
pub fn pack(fmt: &[u8], args: &[PackArg<'_>]) -> Result<Vec<u8>, String> {
    let mut h = header();
    let mut f = Format { fmt, at: 0 };
    let mut out = Vec::new();
    let mut total = 0usize;
    let mut arg = 1usize;
    // An argument not there reads as nil, which is what the message
    // says.
    let missing = |n: usize| arg_error(n, "number expected, got nil");
    let missing_string = |n: usize| arg_error(n, "string expected, got nil");
    while f.peek().is_some() {
        let (opt, size, to_align) = f.details(&mut h, total)?;
        total += to_align + size;
        out.resize(out.len() + to_align, PADBYTE);
        arg += 1;
        let value = args.get(arg - 2);
        match opt {
            Option_::Int => {
                let n = int_of(value.ok_or_else(|| missing(arg))?, arg)?;
                if size < SZINT {
                    let lim = 1i64 << (size * 8 - 1);
                    if !(-lim <= n && n < lim) {
                        return Err(arg_error(arg, "integer overflow"));
                    }
                }
                pack_int(&mut out, n as u64, h.little, size, n < 0);
            }
            Option_::Uint => {
                let n = int_of(value.ok_or_else(|| missing(arg))?, arg)?;
                if size < SZINT && (n as u64) >= (1u64 << (size * 8)) {
                    return Err(arg_error(arg, "unsigned overflow"));
                }
                pack_int(&mut out, n as u64, h.little, size, false);
            }
            Option_::Float => {
                let v = float_of(value.ok_or_else(|| missing(arg))?, arg)? as f32;
                put_bytes(&mut out, &v.to_le_bytes(), h.little);
            }
            Option_::Number | Option_::Double => {
                let v = float_of(value.ok_or_else(|| missing(arg))?, arg)?;
                put_bytes(&mut out, &v.to_le_bytes(), h.little);
            }
            Option_::Char => {
                let a = value.ok_or_else(|| missing_string(arg))?;
                let s = bytes_of(a, arg)?;
                if s.len() > size {
                    return Err(arg_error(arg, "string longer than given size"));
                }
                out.extend_from_slice(&s);
                out.resize(out.len() + (size - s.len()), PADBYTE);
            }
            Option_::String => {
                let a = value.ok_or_else(|| missing_string(arg))?;
                let s = bytes_of(a, arg)?;
                if size < SZINT && (s.len() as u64) >= (1u64 << (size * 8)) {
                    return Err(arg_error(arg, "string length does not fit in given size"));
                }
                pack_int(&mut out, s.len() as u64, h.little, size, false);
                out.extend_from_slice(&s);
                total += s.len();
            }
            Option_::Zstr => {
                let a = value.ok_or_else(|| missing_string(arg))?;
                let s = bytes_of(a, arg)?;
                if s.contains(&0) {
                    return Err(arg_error(arg, "string contains zeros"));
                }
                out.extend_from_slice(&s);
                out.push(0);
                total += s.len() + 1;
            }
            Option_::Padding => {
                out.push(PADBYTE);
                arg -= 1;
            }
            Option_::Paddalign | Option_::Nop => arg -= 1,
        }
    }
    Ok(out)
}

/// `string.packsize(fmt)`.
pub fn packsize(fmt: &[u8]) -> Result<i64, String> {
    let mut h = header();
    let mut f = Format { fmt, at: 0 };
    let mut total = 0usize;
    while f.peek().is_some() {
        let (opt, size, to_align) = f
            .details(&mut h, total)
            .map_err(|e| e.replace("'pack'", "'packsize'"))?;
        if matches!(opt, Option_::String | Option_::Zstr) {
            return Err(arg_error(1, "variable-length format").replace("'pack'", "'packsize'"));
        }
        let size = size + to_align;
        if total > MAXSIZE as usize - size {
            return Err(arg_error(1, "format result too large").replace("'pack'", "'packsize'"));
        }
        total += size;
    }
    Ok(total as i64)
}

fn unpack_int(data: &[u8], little: bool, size: usize, signed: bool) -> Result<i64, String> {
    let limit = size.min(SZINT);
    let mut res: u64 = 0;
    for i in (0..limit).rev() {
        res <<= 8;
        res |= u64::from(data[if little { i } else { size - 1 - i }]);
    }
    if size < SZINT {
        if signed {
            let mask = 1u64 << (size * 8 - 1);
            res = (res ^ mask).wrapping_sub(mask);
        }
    } else if size > SZINT {
        let mask = if !signed || (res as i64) >= 0 {
            0
        } else {
            0xff
        };
        for i in limit..size {
            if data[if little { i } else { size - 1 - i }] != mask {
                return Err(format!("{size}-byte integer does not fit into Lua Integer"));
            }
        }
    }
    Ok(res as i64)
}

fn take_bytes(data: &[u8], little: bool) -> [u8; 8] {
    let mut buf = [0u8; 8];
    let n = data.len().min(8);
    for i in 0..n {
        buf[i] = data[if little { i } else { n - 1 - i }];
    }
    buf
}

/// `string.unpack(fmt, s, pos)`: the values and the position after
/// them; `pos` is already a byte offset from the start.
pub fn unpack(fmt: &[u8], data: &[u8], pos: usize) -> Result<(Vec<Unpacked>, usize), String> {
    let ld = data.len();
    let short = || "bad argument #2 to 'unpack' (data string too short)".to_string();
    if pos > ld {
        return Err("bad argument #3 to 'unpack' (initial position out of string)".to_string());
    }
    let mut h = header();
    let mut f = Format { fmt, at: 0 };
    let mut out = Vec::new();
    let mut pos = pos;
    while f.peek().is_some() {
        let (opt, size, to_align) = f
            .details(&mut h, pos)
            .map_err(|e| e.replace("'pack'", "'unpack'"))?;
        if to_align + size > ld - pos {
            return Err(short());
        }
        pos += to_align;
        match opt {
            Option_::Int | Option_::Uint => {
                let v = unpack_int(
                    &data[pos..pos + size],
                    h.little,
                    size,
                    matches!(opt, Option_::Int),
                )?;
                out.push(Unpacked::Int(v));
            }
            Option_::Float => {
                let buf = take_bytes(&data[pos..pos + 4], h.little);
                let v = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
                out.push(Unpacked::Float(f64::from(v)));
            }
            Option_::Number | Option_::Double => {
                let buf = take_bytes(&data[pos..pos + 8], h.little);
                out.push(Unpacked::Float(f64::from_le_bytes(buf)));
            }
            Option_::Char => out.push(Unpacked::Str(data[pos..pos + size].to_vec())),
            Option_::String => {
                let len = unpack_int(&data[pos..pos + size], h.little, size, false)? as usize;
                if len > ld - pos - size {
                    return Err(short());
                }
                out.push(Unpacked::Str(data[pos + size..pos + size + len].to_vec()));
                pos += len;
            }
            Option_::Zstr => {
                let len = data[pos..].iter().position(|&b| b == 0).unwrap_or(ld - pos);
                if pos + len >= ld {
                    return Err(
                        "bad argument #2 to 'unpack' (unfinished string for format 'z')"
                            .to_string(),
                    );
                }
                out.push(Unpacked::Str(data[pos..pos + len].to_vec()));
                pos += len + 1;
            }
            Option_::Paddalign | Option_::Padding | Option_::Nop => {}
        }
        pos += size;
    }
    Ok((out, pos + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_round_trip() {
        let packed = pack(b"<i4", &[PackArg::Int(-2)]).unwrap();
        assert_eq!(packed, vec![0xfe, 0xff, 0xff, 0xff]);
        let (values, next) = unpack(b"<i4", &packed, 0).unwrap();
        assert_eq!(values, vec![Unpacked::Int(-2)]);
        assert_eq!(next, 5);
        let big = pack(b">i16", &[PackArg::Int(-1)]).unwrap();
        assert_eq!(big, vec![0xff; 16]);
        assert_eq!(unpack(b">i16", &big, 0).unwrap().0, vec![Unpacked::Int(-1)]);
    }

    #[test]
    fn sizes_and_alignment() {
        assert_eq!(packsize(b"!8 b d").unwrap(), 16);
        assert_eq!(packsize(b"!4 b d").unwrap(), 12);
        assert_eq!(packsize(b"b Xd").unwrap(), 1);
        assert_eq!(packsize(b"! b Xd").unwrap(), 8);
        assert!(packsize(b"s").is_err());
        assert!(pack(b"!3 d", &[PackArg::Float(1.0)]).is_err());
    }

    #[test]
    fn strings() {
        let packed = pack(b"z s1", &[PackArg::Str(b"ab"), PackArg::Str(b"xyz")]).unwrap();
        assert_eq!(packed, b"ab\0\x03xyz");
        let (values, next) = unpack(b"z s1", &packed, 0).unwrap();
        assert_eq!(
            values,
            vec![
                Unpacked::Str(b"ab".to_vec()),
                Unpacked::Str(b"xyz".to_vec())
            ]
        );
        assert_eq!(next, 8);
        assert!(pack(b"c2", &[PackArg::Str(b"abc")]).is_err());
        assert!(unpack(b"z", b"abc", 0).is_err());
    }

    #[test]
    fn overflow_is_refused() {
        assert!(pack(b"b", &[PackArg::Int(128)]).is_err());
        assert!(pack(b"B", &[PackArg::Int(256)]).is_err());
        assert!(unpack(b"i9", &[0, 0, 0, 0, 0, 0, 0, 0, 1], 0).is_err());
    }
}
