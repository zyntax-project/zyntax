//! What the host hands a program natively: number formatting and
//! parsing as Lua does them, `string.format`, the clocks, and the
//! arguments the program was started with. Reached from the library
//! as `$Lua$…` symbols.

use std::sync::OnceLock;

use zrtl::{DynamicBox, StringPtr, TypeCategory, TypeTag};

use crate::{host_io, host_os};

static ARGS: OnceLock<Vec<String>> = OnceLock::new();

/// The program's arguments, `arg` in Lua's terms: the interpreter,
/// then the script's path, then the rest. Set once per process; a
/// later call keeps the first.
pub fn set_args(args: Vec<String>) {
    let _ = ARGS.set(args);
}

extern "C" fn host_argc() -> i64 {
    ARGS.get().map_or(0, |args| args.len() as i64)
}

extern "C" fn host_argv(i: i64) -> StringPtr {
    match usize::try_from(i)
        .ok()
        .and_then(|i| ARGS.get().and_then(|args| args.get(i)))
    {
        Some(arg) => zrtl::string_new(arg),
        None => std::ptr::null_mut(),
    }
}

/// Seconds of CPU time, as `os.clock` reports.
extern "C" fn host_clock() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Seconds since the epoch, as `os.time()` reports.
extern "C" fn host_time() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─── numbers ────────────────────────────────────────────────────────

/// A float as `%.14g` prints it, with `.0` appended when the digits
/// alone would read as an integer, as Lua's `tostring` does.
pub fn float_text(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let s = format_g(x, 14);
    if s.bytes().all(|b| b.is_ascii_digit() || b == b'-') {
        format!("{s}.0")
    } else {
        s
    }
}

/// C's `%.<p>g`: `p` significant digits, positional when the exponent
/// is in `[-4, p)`, scientific otherwise, trailing zeros dropped.
pub fn format_g(x: f64, p: usize) -> String {
    format_g_flags(x, p, false, false)
}

/// `%g` with the `#` flag (keep trailing zeros) and upper-case `E`.
fn format_g_flags(x: f64, p: usize, alt: bool, upper: bool) -> String {
    let p = p.max(1);
    if x == 0.0 {
        let zero = if x.is_sign_negative() { "-0" } else { "0" };
        return if alt {
            format!("{zero}.{}", "0".repeat(p - 1))
        } else {
            zero.to_string()
        };
    }
    // The exponent after rounding to p digits comes from the
    // scientific form itself.
    let sci = format!("{:.*e}", p - 1, x);
    let (mantissa, exp) = sci.split_once('e').expect("an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let mut out = if exp >= -4 && exp < p as i32 {
        format!("{:.*}", (p as i32 - 1 - exp).max(0) as usize, x)
    } else {
        let e = if upper { 'E' } else { 'e' };
        format!(
            "{mantissa}{e}{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    };
    if !alt {
        out = strip_zeros(&out);
    }
    out
}

/// Trailing zeros of a fraction, and a bare point, carry nothing; the
/// exponent, if any, stays.
fn strip_zeros(s: &str) -> String {
    let (body, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    };
    if !body.contains('.') {
        return s.to_string();
    }
    let body = body.trim_end_matches('0').trim_end_matches('.');
    format!("{body}{exp}")
}

extern "C" fn host_float_str(x: f64) -> StringPtr {
    zrtl::string_new(&float_text(x))
}

/// What a string converts to as a number: nothing, an integer, or a
/// float, by Lua's rules for numerals: surrounding whitespace, a sign,
/// decimal or hexadecimal digits, a fraction and an exponent. A
/// decimal integer that does not fit is a float; a hexadecimal one
/// wraps.
pub enum Numeral {
    None,
    Int(i64),
    Float(f64),
}

pub fn parse_numeral(s: &str) -> Numeral {
    let s = s.trim_matches(|c: char| c == ' ' || ('\t'..='\r').contains(&c));
    if s.is_empty() {
        return Numeral::None;
    }
    let (negative, body) = match s.as_bytes()[0] {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if body.is_empty() {
        return Numeral::None;
    }
    let lower = body.to_ascii_lowercase();
    let result = if let Some(hex) = lower.strip_prefix("0x") {
        parse_hex(hex)
    } else {
        parse_decimal(&lower, negative)
    };
    match result {
        Numeral::Int(v) if negative => Numeral::Int(v.wrapping_neg()),
        Numeral::Float(v) if negative => Numeral::Float(-v),
        other => other,
    }
}

fn parse_decimal(s: &str, negative: bool) -> Numeral {
    let bytes = s.as_bytes();
    let is_float = bytes.iter().any(|b| matches!(b, b'.' | b'e'));
    if !is_float {
        if !bytes.iter().all(u8::is_ascii_digit) {
            return Numeral::None;
        }
        // The magnitude may reach 2^63 when the sign brings it back
        // into range; anything larger is the float it rounds to.
        let limit: u64 = if negative {
            1u64 << 63
        } else {
            i64::MAX as u64
        };
        return match s.parse::<u64>() {
            Ok(v) if v <= limit => Numeral::Int(v as i64),
            _ => match s.parse::<f64>() {
                Ok(v) => Numeral::Float(v),
                Err(_) => Numeral::None,
            },
        };
    }
    // Rust's parser takes what Lua takes, except that Lua allows a
    // fraction with no digits before or after the point but never a
    // bare point, and Rust refuses "1." and ".5" nowhere; both agree.
    if s == "." || s.starts_with('e') || s.ends_with('e') {
        return Numeral::None;
    }
    let valid = bytes
        .iter()
        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-'));
    if !valid {
        return Numeral::None;
    }
    match s.parse::<f64>() {
        Ok(v) => Numeral::Float(v),
        Err(_) => Numeral::None,
    }
}

fn parse_hex(s: &str) -> Numeral {
    let (mantissa, exp) = match s.split_once('p') {
        Some((m, e)) => (m, Some(e)),
        None => (s, None),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (mantissa, None),
    };
    let hex_digit = |b: u8| (b as char).to_digit(16);
    if int_part.is_empty() && frac_part.is_none_or(str::is_empty) {
        return Numeral::None;
    }
    if !int_part.bytes().all(|b| hex_digit(b).is_some()) {
        return Numeral::None;
    }
    if frac_part.is_none() && exp.is_none() {
        // A hexadecimal integer wraps around.
        let mut v: u64 = 0;
        for b in int_part.bytes() {
            v = v
                .wrapping_mul(16)
                .wrapping_add(hex_digit(b).unwrap() as u64);
        }
        return Numeral::Int(v as i64);
    }
    // The digits go into the mantissa while they are significant and
    // fit; past that they only move the binary exponent, as the
    // reference reads them, so any number of digits is read exactly.
    const MAX_SIGNIFICANT: u32 = 30;
    if frac_part.is_some_and(|f| !f.bytes().all(|b| hex_digit(b).is_some())) {
        return Numeral::None;
    }
    let mut r = 0f64;
    let mut e: i64 = 0;
    let mut significant = 0u32;
    let mut take = |b: u8, after_point: bool| {
        let digit = hex_digit(b).unwrap();
        if significant == 0 && digit == 0 {
            // A leading zero is not significant.
        } else {
            significant += 1;
            if significant <= MAX_SIGNIFICANT {
                r = r * 16.0 + digit as f64;
            } else {
                e += 1;
            }
        }
        if after_point {
            e -= 1;
        }
    };
    for b in int_part.bytes() {
        take(b, false);
    }
    for b in frac_part.unwrap_or("").bytes() {
        take(b, true);
    }
    e *= 4;
    if let Some(exp) = exp {
        let Ok(p) = exp.parse::<i64>() else {
            return Numeral::None;
        };
        e += p;
    }
    Numeral::Float(ldexp(r, e))
}

/// `r * 2^e`, scaled in steps so a large exponent on a mantissa that
/// brings it back into range does not overflow on the way.
fn ldexp(mut r: f64, mut e: i64) -> f64 {
    while e > 1000 {
        r *= 2f64.powi(1000);
        e -= 1000;
    }
    while e < -1000 {
        r *= 2f64.powi(-1000);
        e += 1000;
    }
    r * 2f64.powi(e as i32)
}

unsafe fn text_of(s: zrtl::StringConstPtr) -> &'static str {
    unsafe { zrtl::string_as_str(s) }.unwrap_or("")
}

/// 0 when the text is no numeral, 1 for an integer, 2 for a float.
extern "C" fn host_number_kind(s: zrtl::StringConstPtr) -> i64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::None => 0,
        Numeral::Int(_) => 1,
        Numeral::Float(_) => 2,
    }
}
extern "C" fn host_parse_int(s: zrtl::StringConstPtr) -> i64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::Int(v) => v,
        _ => 0,
    }
}
extern "C" fn host_parse_float(s: zrtl::StringConstPtr) -> f64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::Float(v) => v,
        Numeral::Int(v) => v as f64,
        Numeral::None => 0.0,
    }
}

/// Whether a float has an exact integer value that fits.
extern "C" fn host_float_is_int(x: f64) -> bool {
    x.fract() == 0.0 && (-9223372036854775808.0..9223372036854775808.0).contains(&x)
}

// ─── string.format ──────────────────────────────────────────────────

/// A list of dynamic values as the compiler lays it out: the header
/// holds the data pointer, the length and the capacity.
#[repr(C)]
struct ListHeader {
    data: *const *const DynamicBox,
    len: i64,
    capacity: i64,
}

/// A dynamic value read for formatting. A string is its bytes, any
/// of them; a string or an object carries its address, for `%p`. An
/// object's text is what `tostring` gives it, metamethods included,
/// which the library works out before the call.
pub(crate) enum Arg<'a> {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a [u8], usize),
    Other(Vec<u8>, usize),
}

unsafe fn read_arg<'a>(b: *const DynamicBox) -> Arg<'a> {
    if b.is_null() {
        return Arg::Nil;
    }
    let b = unsafe { &*b };
    match b.tag.category() {
        TypeCategory::Void => Arg::Nil,
        TypeCategory::Bool => Arg::Bool(b.as_bool().unwrap_or(false)),
        TypeCategory::Int | TypeCategory::UInt => {
            if b.tag == TypeTag::I64 {
                Arg::Int(b.as_i64().unwrap_or(0))
            } else {
                Arg::Int(b.as_i32().map(i64::from).unwrap_or(0))
            }
        }
        TypeCategory::Float => Arg::Float(b.as_f64().unwrap_or(0.0)),
        TypeCategory::String => Arg::Str(
            unsafe { bytes_of(b.data as zrtl::StringConstPtr) },
            b.data as usize,
        ),
        _ if type_word(b) == "FILE*" => Arg::Other(
            host_io::file_text(b.data as i64).into_bytes(),
            b.data as usize,
        ),
        _ => Arg::Other(
            format!("{}: 0x{:x}", type_word(b), b.data as usize).into_bytes(),
            b.data as usize,
        ),
    }
}

fn type_word(b: &DynamicBox) -> &'static str {
    let kind = b.tag.raw() >> 8;
    if kind == (zyntax_builtins::FUNC_TAG >> 8) as u32 {
        "function"
    } else if kind == zyntax_builtins::instance_tag(super::library::THREAD_KIND) as u32 >> 8 {
        "thread"
    } else if kind == zyntax_builtins::instance_tag(super::library::FILE_KIND) as u32 >> 8 {
        "FILE*"
    } else {
        "table"
    }
}

/// `string.format(fmt, ...)`: C's conversions with Lua's additions
/// (`%q`), over bytes, or the message of the error to raise.
pub fn format(fmt: &[u8], args: &[Arg<'_>]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let bytes = fmt;
    let mut i = 0;
    let mut next = 0;
    let take = |next: &mut usize| -> Result<&Arg<'_>, String> {
        let a = args
            .get(*next)
            .ok_or_else(|| format!("bad argument #{} to 'format' (no value)", *next + 2))?;
        *next += 1;
        Ok(a)
    };
    while i < bytes.len() {
        if bytes[i] != b'%' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'%' {
                i += 1;
            }
            out.extend_from_slice(&fmt[start..i]);
            continue;
        }
        i += 1;
        if i >= bytes.len() {
            return Err("invalid conversion '%' to 'format'".to_string());
        }
        if bytes[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        // Flags, width, precision.
        let spec_start = i;
        while i < bytes.len() && matches!(bytes[i], b'-' | b'+' | b' ' | b'#' | b'0') {
            i += 1;
        }
        let flags = String::from_utf8_lossy(&fmt[spec_start..i]).into_owned();
        let width_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let width: Option<usize> = String::from_utf8_lossy(&fmt[width_start..i]).parse().ok();
        let mut precision: Option<usize> = None;
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            let p_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            precision = Some(
                String::from_utf8_lossy(&fmt[p_start..i])
                    .parse()
                    .unwrap_or(0),
            );
        }
        if i >= bytes.len() {
            return Err(format!(
                "invalid conversion '%{}' to 'format'",
                String::from_utf8_lossy(&fmt[spec_start..])
            ));
        }
        let conv = bytes[i] as char;
        i += 1;
        // The reference bounds a specification and, per conversion,
        // which flags it may carry and how wide its width and
        // precision may be.
        let spec = &fmt[spec_start..i - 1];
        if spec.len() + 1 >= 22 {
            return Err("invalid format (too long)".to_string());
        }
        let allowed: &[u8] = match conv {
            'c' | 'p' | 's' => b"-",
            'd' | 'i' => b"-+ 0",
            'u' => b"-0",
            'o' | 'x' | 'X' => b"-#0",
            'a' | 'A' | 'f' | 'e' | 'E' | 'g' | 'G' => b"-+ #0",
            'q' => {
                if !spec.is_empty() {
                    return Err("specifier '%q' cannot have modifiers".to_string());
                }
                b""
            }
            _ => b"",
        };
        if conv != 'q' && !spec_is_valid(spec, allowed, !matches!(conv, 'c' | 'p')) {
            return Err(format!(
                "invalid conversion specification: '%{}{conv}'",
                String::from_utf8_lossy(spec)
            ));
        }
        let left = flags.contains('-');
        let zero = flags.contains('0') && !left;
        let plus = flags.contains('+');
        let space = flags.contains(' ');
        let alt = flags.contains('#');
        let arg_no = next + 2;
        let body: Vec<u8> = match conv {
            'd' | 'i' => {
                let v = int_arg(take(&mut next)?, arg_no)?;
                let digits = v.unsigned_abs().to_string();
                let digits = match precision {
                    Some(p) if digits.len() < p => {
                        format!("{}{digits}", "0".repeat(p - digits.len()))
                    }
                    Some(0) if v == 0 => String::new(),
                    _ => digits,
                };
                signed(digits, v < 0, plus, space).into_bytes()
            }
            'u' => {
                let v = int_arg(take(&mut next)?, arg_no)?;
                let digits = (v as u64).to_string();
                match precision {
                    Some(p) if digits.len() < p => {
                        format!("{}{digits}", "0".repeat(p - digits.len()))
                    }
                    Some(0) if v == 0 => String::new(),
                    _ => digits,
                }
                .into_bytes()
            }
            'c' => {
                let v = int_arg(take(&mut next)?, arg_no)?;
                vec![v as u8]
            }
            'x' | 'X' | 'o' => {
                let v = int_arg(take(&mut next)?, arg_no)? as u64;
                let digits = match conv {
                    'x' => format!("{v:x}"),
                    'X' => format!("{v:X}"),
                    _ => format!("{v:o}"),
                };
                let digits = match precision {
                    Some(p) if digits.len() < p => {
                        format!("{}{digits}", "0".repeat(p - digits.len()))
                    }
                    Some(0) if v == 0 && !(alt && conv == 'o') => String::new(),
                    _ => digits,
                };
                if alt && v != 0 {
                    match conv {
                        'x' => format!("0x{digits}"),
                        'X' => format!("0X{digits}"),
                        _ => format!("0{digits}"),
                    }
                } else {
                    digits
                }
                .into_bytes()
            }
            // `%F` is not a conversion the reference takes.
            'f' | 'e' | 'E' | 'g' | 'G' => {
                let v = float_arg(take(&mut next)?, arg_no)?;
                let p = precision.unwrap_or(6);
                if v.is_nan() {
                    "nan".to_string()
                } else if v.is_infinite() {
                    signed("inf".to_string(), v < 0.0, plus, space)
                } else {
                    let mag = v.abs();
                    let digits = match conv {
                        // `#` keeps the point even with nothing after it.
                        'f' if alt && p == 0 => format!("{mag:.0}."),
                        'f' => format!("{mag:.p$}"),
                        'e' | 'E' => {
                            let s = format!("{:.*e}", p, mag);
                            let (m, e) = s.split_once('e').expect("an exponent");
                            let e: i32 = e.parse().expect("an integer exponent");
                            let e_char = if conv == 'E' { 'E' } else { 'e' };
                            format!("{m}{e_char}{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
                        }
                        _ => format_g_flags(mag, p, alt, conv == 'G'),
                    };
                    signed(
                        digits,
                        v.is_sign_negative() && v != 0.0 || v < 0.0,
                        plus,
                        space,
                    )
                }
                .into_bytes()
            }
            'a' | 'A' => {
                let v = float_arg(take(&mut next)?, arg_no)?;
                let text = hex_float_digits(v, conv == 'A', precision);
                let negative = text.starts_with('-');
                let magnitude = text.trim_start_matches('-').to_string();
                signed(magnitude, negative, plus, space).into_bytes()
            }
            's' => {
                let a = take(&mut next)?;
                let s: Vec<u8> = match a {
                    Arg::Nil => b"nil".to_vec(),
                    Arg::Bool(b) => b.to_string().into_bytes(),
                    Arg::Int(v) => v.to_string().into_bytes(),
                    Arg::Float(v) => float_text(*v).into_bytes(),
                    Arg::Str(s, _) => s.to_vec(),
                    Arg::Other(s, _) => s.clone(),
                };
                // C's conversion stops at a zero byte, so one with a
                // width or precision is refused; plain `%s` copies.
                if (width.is_some() || precision.is_some() || !flags.is_empty()) && s.contains(&0) {
                    return Err(format!(
                        "bad argument #{arg_no} to 'format' (string contains zeros)"
                    ));
                }
                match precision {
                    Some(p) => s.into_iter().take(p).collect(),
                    None => s,
                }
            }
            // The address of what is allocated; nothing for a value
            // that is not, printed as C prints a null pointer.
            'p' => match take(&mut next)? {
                Arg::Str(_, address) | Arg::Other(_, address) => {
                    format!("0x{address:x}").into_bytes()
                }
                _ => b"(null)".to_vec(),
            },
            'q' => {
                let a = take(&mut next)?;
                match a {
                    Arg::Str(s, _) => quoted(s),
                    // The least integer has no decimal literal that reads
                    // back as an integer; it goes in hexadecimal.
                    Arg::Int(i64::MIN) => b"0x8000000000000000".to_vec(),
                    Arg::Int(v) => v.to_string().into_bytes(),
                    // A float reads back exactly in hexadecimal; the
                    // infinities and NaN have no literal and are spelled
                    // as expressions that give them.
                    Arg::Float(v) => if v.is_infinite() {
                        if *v > 0.0 { "1e9999" } else { "-1e9999" }.to_string()
                    } else if v.is_nan() {
                        "(0/0)".to_string()
                    } else {
                        hex_float(*v, false)
                    }
                    .into_bytes(),
                    Arg::Nil => b"nil".to_vec(),
                    Arg::Bool(b) => b.to_string().into_bytes(),
                    Arg::Other(..) => {
                        return Err(format!(
                            "bad argument #{arg_no} to 'format' (value has no literal form)"
                        ));
                    }
                }
            }
            other => {
                return Err(format!("invalid conversion '%{other}' to 'format'"));
            }
        };
        // Width, in bytes as C counts it, padding with zeros for
        // numbers when asked and nothing says otherwise.
        let padded = match width {
            Some(w) if body.len() < w => {
                let pad = w - body.len();
                let mut padded = Vec::with_capacity(w);
                if left {
                    padded.extend_from_slice(&body);
                    padded.resize(w, b' ');
                } else if zero
                    && matches!(
                        conv,
                        'd' | 'i' | 'u' | 'x' | 'X' | 'o' | 'f' | 'e' | 'E' | 'g' | 'G'
                    )
                {
                    let sign = match body.first() {
                        Some(b'-' | b'+' | b' ') => 1,
                        _ => 0,
                    };
                    padded.extend_from_slice(&body[..sign]);
                    padded.resize(sign + pad, b'0');
                    padded.extend_from_slice(&body[sign..]);
                } else {
                    padded.resize(pad, b' ');
                    padded.extend_from_slice(&body);
                }
                padded
            }
            _ => body,
        };
        out.extend_from_slice(&padded);
    }
    Ok(out)
}

fn signed(digits: String, negative: bool, plus: bool, space: bool) -> String {
    if negative {
        format!("-{digits}")
    } else if plus {
        format!("+{digits}")
    } else if space {
        format!(" {digits}")
    } else {
        digits
    }
}

fn int_arg(a: &Arg<'_>, n: usize) -> Result<i64, String> {
    match a {
        Arg::Int(v) => Ok(*v),
        Arg::Float(v) if v.fract() == 0.0 => Ok(*v as i64),
        Arg::Float(_) => Err(format!(
            "bad argument #{n} to 'format' (number has no integer representation)"
        )),
        Arg::Str(s, _) => match parse_numeral(&String::from_utf8_lossy(s)) {
            Numeral::Int(v) => Ok(v),
            Numeral::Float(v) if v.fract() == 0.0 => Ok(v as i64),
            _ => Err(format!(
                "bad argument #{n} to 'format' (number expected, got string)"
            )),
        },
        other => Err(format!(
            "bad argument #{n} to 'format' (number expected, got {})",
            arg_type(other)
        )),
    }
}

fn float_arg(a: &Arg<'_>, n: usize) -> Result<f64, String> {
    match a {
        Arg::Int(v) => Ok(*v as f64),
        Arg::Float(v) => Ok(*v),
        Arg::Str(s, _) => match parse_numeral(&String::from_utf8_lossy(s)) {
            Numeral::Int(v) => Ok(v as f64),
            Numeral::Float(v) => Ok(v),
            Numeral::None => Err(format!(
                "bad argument #{n} to 'format' (number expected, got string)"
            )),
        },
        other => Err(format!(
            "bad argument #{n} to 'format' (number expected, got {})",
            arg_type(other)
        )),
    }
}

fn arg_type(a: &Arg<'_>) -> &'static str {
    match a {
        Arg::Nil => "nil",
        Arg::Bool(_) => "boolean",
        Arg::Int(_) | Arg::Float(_) => "number",
        Arg::Str(..) => "string",
        Arg::Other(..) => "table",
    }
}

/// Whether a conversion's flags, width and precision are ones the
/// reference takes: flags from `allowed`, a width of up to two
/// digits not starting with zero (that is a flag), and a precision
/// of up to two digits where one is allowed.
fn spec_is_valid(spec: &[u8], allowed: &[u8], precision: bool) -> bool {
    let mut s = spec;
    while let Some((&b, rest)) = s.split_first()
        && allowed.contains(&b)
    {
        s = rest;
    }
    fn digits(s: &[u8]) -> &[u8] {
        let n = s.iter().take(2).take_while(|b| b.is_ascii_digit()).count();
        &s[n..]
    }
    if s.first() != Some(&b'0') {
        s = digits(s);
        if s.first() == Some(&b'.') && precision {
            s = digits(&s[1..]);
        }
    }
    s.is_empty()
}

/// `%q`: the string as a Lua literal that reads back as itself: a
/// control byte is a decimal escape, three digits wide when a digit
/// follows; any other byte is itself.
fn quoted(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\\n"),
            b if b < 0x20 || b == 0x7f => {
                let escape = if bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
                    format!("\\{b:03}")
                } else {
                    format!("\\{b}")
                };
                out.extend_from_slice(escape.as_bytes());
            }
            b => out.push(b),
        }
    }
    out.push(b'"');
    out
}

/// `%a`: a float in hexadecimal, as C prints it.
fn hex_float(v: f64, upper: bool) -> String {
    hex_float_digits(v, upper, None)
}

/// `%.Na`: the mantissa rounded to `precision` hexadecimal digits,
/// to the nearest with a tie going toward zero, as the C library
/// here rounds it; without one, exactly, with no trailing zeros.
fn hex_float_digits(v: f64, upper: bool, precision: Option<usize>) -> String {
    let s = if v.is_nan() {
        "nan".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.to_string()
    } else if v == 0.0 {
        let sign = if v.is_sign_negative() { "-" } else { "" };
        match precision {
            Some(p) if p > 0 => format!("{sign}0x0.{}p+0", "0".repeat(p)),
            _ => format!("{sign}0x0p+0"),
        }
    } else {
        let bits = v.to_bits();
        let sign = if bits >> 63 == 1 { "-" } else { "" };
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let mant = bits & ((1u64 << 52) - 1);
        let (mut lead, exp) = if exp == 0 {
            (0u64, -1022)
        } else {
            (1u64, exp - 1023)
        };
        let digits = match precision {
            Some(p) if p < 13 => {
                let shift = 52 - 4 * p as u32;
                let whole = (lead << 52) | mant;
                let half = 1u64 << (shift - 1);
                let low = whole & ((1u64 << shift) - 1);
                let mut rounded = whole >> shift;
                if low > half {
                    rounded += 1;
                }
                lead = rounded >> (4 * p);
                let frac = rounded & ((1u64 << (4 * p)) - 1);
                if p == 0 {
                    String::new()
                } else {
                    format!("{frac:0width$x}", width = p)
                }
            }
            Some(p) => format!("{mant:013x}{}", "0".repeat(p - 13)),
            None => format!("{mant:013x}").trim_end_matches('0').to_string(),
        };
        if digits.is_empty() {
            format!("{sign}0x{lead}p{exp:+}")
        } else {
            format!("{sign}0x{lead}.{digits}p{exp:+}")
        }
    };
    if upper { s.to_uppercase() } else { s }
}

unsafe fn args_of<'a>(list: *const ListHeader) -> Vec<Arg<'a>> {
    if list.is_null() {
        return Vec::new();
    }
    let header = unsafe { &*list };
    let len = header.len.max(0) as usize;
    (0..len)
        .map(|i| unsafe { read_arg(*header.data.add(i)) })
        .collect()
}

/// The arguments, each object's text taken from `texts` (a string
/// there, nil for any other value).
unsafe fn args_with_texts<'a>(list: *const ListHeader, texts: *const ListHeader) -> Vec<Arg<'a>> {
    let mut args = unsafe { args_of(list) };
    if texts.is_null() {
        return args;
    }
    let header = unsafe { &*texts };
    let len = (header.len.max(0) as usize).min(args.len());
    for (i, arg) in args.iter_mut().enumerate().take(len) {
        if let Arg::Other(text, _) = arg {
            let b = unsafe { *header.data.add(i) };
            if !b.is_null()
                && let TypeCategory::String = unsafe { (*b).tag.category() }
            {
                *text = unsafe { bytes_of((*b).data as zrtl::StringConstPtr) }.to_vec();
            }
        }
    }
    args
}

/// The formatted text, or the error message prefixed by a byte no text
/// starts with, for the library to raise.
extern "C" fn host_format(
    fmt: zrtl::StringConstPtr,
    args: *const ListHeader,
    texts: *const ListHeader,
) -> StringPtr {
    let fmt = unsafe { bytes_of(fmt) };
    let args = unsafe { args_with_texts(args, texts) };
    match format(fmt, &args) {
        Ok(s) => zrtl::string::string_from_bytes(&s),
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            std::ptr::null_mut()
        }
    }
}

// ─── string.pack ────────────────────────────────────────────────────

thread_local! {
    static PACK_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static UNPACKED: std::cell::RefCell<(Vec<crate::pack::Unpacked>, usize)> = const { std::cell::RefCell::new((Vec::new(), 0)) };
}

/// One argument as `string.pack` takes it.
unsafe fn read_pack_arg<'a>(b: *const DynamicBox) -> crate::pack::PackArg<'a> {
    use crate::pack::PackArg;
    if b.is_null() {
        return PackArg::Other("nil");
    }
    let b = unsafe { &*b };
    match b.tag.category() {
        TypeCategory::Void => PackArg::Other("nil"),
        TypeCategory::Bool => PackArg::Other("boolean"),
        TypeCategory::Int | TypeCategory::UInt => {
            if b.tag == TypeTag::I64 {
                PackArg::Int(b.as_i64().unwrap_or(0))
            } else {
                PackArg::Int(b.as_i32().map(i64::from).unwrap_or(0))
            }
        }
        TypeCategory::Float => PackArg::Float(b.as_f64().unwrap_or(0.0)),
        TypeCategory::String => PackArg::Str(unsafe { bytes_of(b.data as zrtl::StringConstPtr) }),
        _ => PackArg::Other(type_word(b)),
    }
}

/// The packed bytes, or null with the message kept for `pack_error`.
extern "C" fn host_pack(fmt: zrtl::StringConstPtr, args: *const ListHeader) -> StringPtr {
    let fmt = unsafe { bytes_of(fmt) };
    let args: Vec<crate::pack::PackArg<'_>> = if args.is_null() {
        Vec::new()
    } else {
        let header = unsafe { &*args };
        let len = header.len.max(0) as usize;
        (0..len)
            .map(|i| unsafe { read_pack_arg(*header.data.add(i)) })
            .collect()
    };
    match crate::pack::pack(fmt, &args) {
        Ok(bytes) => zrtl::string::string_from_bytes(&bytes),
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            std::ptr::null_mut()
        }
    }
}

extern "C" fn host_pack_error() -> StringPtr {
    PACK_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().as_bytes()))
}

/// The size, or -1 with the message kept.
extern "C" fn host_packsize(fmt: zrtl::StringConstPtr) -> i64 {
    match crate::pack::packsize(unsafe { bytes_of(fmt) }) {
        Ok(n) => n,
        Err(e) => {
            PACK_ERROR.with(|err| *err.borrow_mut() = e);
            -1
        }
    }
}

/// How many values were unpacked, kept for the accessors, or -1 with
/// the message kept. `pos` is a byte offset.
extern "C" fn host_unpack(fmt: zrtl::StringConstPtr, s: zrtl::StringConstPtr, pos: i64) -> i64 {
    let (fmt, data) = unsafe { (bytes_of(fmt), bytes_of(s)) };
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

/// 0 for an integer, 1 for a float, 2 for a string.
extern "C" fn host_unpack_kind(i: i64) -> i64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Int(_)) => 0,
        Some(crate::pack::Unpacked::Float(_)) => 1,
        _ => 2,
    })
}

extern "C" fn host_unpack_int(i: i64) -> i64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Int(v)) => *v,
        _ => 0,
    })
}

extern "C" fn host_unpack_float(i: i64) -> f64 {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Float(v)) => *v,
        _ => 0.0,
    })
}

extern "C" fn host_unpack_str(i: i64) -> StringPtr {
    UNPACKED.with(|u| match u.borrow().0.get(i as usize) {
        Some(crate::pack::Unpacked::Str(v)) => zrtl::string::string_from_bytes(v),
        _ => zrtl::string::string_from_bytes(b""),
    })
}

/// The position after the last unpacked value, 1-based.
extern "C" fn host_unpack_next() -> i64 {
    UNPACKED.with(|u| u.borrow().1 as i64)
}

/// `%.14g` of a float for `string.format`'s callers that have one.
extern "C" fn host_format_g(x: f64, precision: i64) -> StringPtr {
    zrtl::string_new(&format_g(x, precision.clamp(1, 99) as usize))
}

/// A word as it is: declared by the library with another type at
/// either end, it reads an address back as the value it points to.
extern "C" fn host_word(w: i64) -> i64 {
    w
}

// ─── strings by the byte ────────────────────────────────────────────

unsafe fn bytes_of(s: zrtl::StringConstPtr) -> &'static [u8] {
    unsafe { zrtl::string_as_bytes(s) }
}

/// Lua's string positions: 1-based, negative from the end, clamped.
/// The range `i..=j` as byte offsets, empty when `i > j`.
fn byte_range(len: usize, i: i64, j: i64) -> (usize, usize) {
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

extern "C" fn host_sub(s: zrtl::StringConstPtr, i: i64, j: i64) -> StringPtr {
    let bytes = unsafe { bytes_of(s) };
    let (a, b) = byte_range(bytes.len(), i, j);
    zrtl::string::string_from_bytes(&bytes[a..b])
}

/// The byte at 1-based position `i`, or -1 outside the string.
extern "C" fn host_byte_at(s: zrtl::StringConstPtr, i: i64) -> i64 {
    let bytes = unsafe { bytes_of(s) };
    let (a, b) = byte_range(bytes.len(), i, i);
    if a < b { bytes[a] as i64 } else { -1 }
}

extern "C" fn host_from_byte(b: i64) -> StringPtr {
    zrtl::string::string_from_bytes(&[b as u8])
}

extern "C" fn host_reverse(s: zrtl::StringConstPtr) -> StringPtr {
    let mut bytes = unsafe { bytes_of(s) }.to_vec();
    bytes.reverse();
    zrtl::string::string_from_bytes(&bytes)
}

/// `string.rep`, or null for a result the reference bounds by a C
/// int: the library raises for it.
extern "C" fn host_rep(s: zrtl::StringConstPtr, n: i64, sep: zrtl::StringConstPtr) -> StringPtr {
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
extern "C" fn host_find_plain(
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
extern "C" fn host_bytes(hex: zrtl::StringConstPtr) -> StringPtr {
    let hex = unsafe { bytes_of(hex) };
    let bytes: Vec<u8> = hex
        .chunks(2)
        .filter_map(|pair| std::str::from_utf8(pair).ok())
        .filter_map(|pair| u8::from_str_radix(pair, 16).ok())
        .collect();
    zrtl::string::string_from_bytes(&bytes)
}

extern "C" fn host_upper(s: zrtl::StringConstPtr) -> StringPtr {
    zrtl::string::string_from_bytes(&unsafe { bytes_of(s) }.to_ascii_uppercase())
}
extern "C" fn host_lower(s: zrtl::StringConstPtr) -> StringPtr {
    zrtl::string::string_from_bytes(&unsafe { bytes_of(s) }.to_ascii_lowercase())
}

// ─── random ─────────────────────────────────────────────────────────

/// xoshiro256**, as Lua 5.4 uses; seeded from the clock unless
/// `math.randomseed` says otherwise.
struct Xoshiro([u64; 4]);

impl Xoshiro {
    fn seeded(n: u64) -> Self {
        Self::seeded_with(n, 0)
    }
    /// The reference's seeding: the two numbers and a constant in the
    /// state, then sixteen draws discarded.
    fn seeded_with(n1: u64, n2: u64) -> Self {
        let mut r = Xoshiro([n1, 0xff, n2, 0]);
        for _ in 0..16 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
}

thread_local! {
    static RANDOM: std::cell::RefCell<Xoshiro> = std::cell::RefCell::new(Xoshiro::seeded(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(7)
    ));
}

extern "C" fn host_random_seed(n1: i64, n2: i64) {
    RANDOM.with(|r| *r.borrow_mut() = Xoshiro::seeded_with(n1 as u64, n2 as u64));
}

/// Seeded from the clock and the process, as `math.randomseed()`
/// with nothing given; the first number, the second being the
/// process id.
extern "C" fn host_random_seed_now() -> i64 {
    let n1 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(7);
    let n2 = u64::from(std::process::id());
    RANDOM.with(|r| *r.borrow_mut() = Xoshiro::seeded_with(n1, n2));
    n1 as i64
}

/// The second number `math.randomseed()` seeds with.
extern "C" fn host_random_seed_pid() -> i64 {
    i64::from(std::process::id())
}

/// The next draw, all 64 bits.
extern "C" fn host_random_next() -> i64 {
    RANDOM.with(|r| r.borrow_mut().next()) as i64
}

/// A float in `[0, 1)`, from the high 53 bits.
extern "C" fn host_random_float() -> f64 {
    let bits = RANDOM.with(|r| r.borrow_mut().next());
    (bits >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// An integer in `[lo, hi]`, projected as the reference projects a
/// draw: masked to the smallest all-ones bound and drawn again while
/// out of range.
extern "C" fn host_random_int(lo: i64, hi: i64) -> i64 {
    if lo > hi {
        return lo;
    }
    let n = (hi as u64).wrapping_sub(lo as u64);
    let mut ran = RANDOM.with(|r| r.borrow_mut().next());
    let pick = if n & n.wrapping_add(1) == 0 {
        ran & n
    } else {
        let mut lim = n;
        lim |= lim >> 1;
        lim |= lim >> 2;
        lim |= lim >> 4;
        lim |= lim >> 8;
        lim |= lim >> 16;
        lim |= lim >> 32;
        loop {
            ran &= lim;
            if ran <= n {
                break ran;
            }
            ran = RANDOM.with(|r| r.borrow_mut().next());
        }
    };
    (lo as u64).wrapping_add(pick) as i64
}

// ─── patterns ───────────────────────────────────────────────────────
// The matcher keeps the last match's captures per thread; the library
// reads them back after a successful call. Errors come back as -2 with
// the message held for `$Lua$pat_error`.

use crate::pattern;

extern "C" fn host_pat_specials(pat: zrtl::StringConstPtr) -> bool {
    pattern::has_specials(unsafe { bytes_of(pat) })
}

/// The first match at or after 0-based `init`: its start, -1 or -2.
extern "C" fn host_pat_find(s: zrtl::StringConstPtr, pat: zrtl::StringConstPtr, init: i64) -> i64 {
    let (src, pat) = unsafe { (bytes_of(s), bytes_of(pat)) };
    pattern::find(src, pat, init as usize)
}

/// A match exactly at 0-based `pos`: its end, -1 or -2.
extern "C" fn host_pat_match_at(
    s: zrtl::StringConstPtr,
    pat: zrtl::StringConstPtr,
    pos: i64,
) -> i64 {
    let (src, pat) = unsafe { (bytes_of(s), bytes_of(pat)) };
    pattern::match_here(src, pat, pos as usize)
}

extern "C" fn host_pat_end() -> i64 {
    pattern::STATE.with(|st| st.borrow().end as i64)
}

extern "C" fn host_pat_level() -> i64 {
    pattern::STATE.with(|st| st.borrow().level as i64)
}

/// Capture `i` of the last match: 0 for text, 1 for a position, -1
/// for an error (held for `$Lua$pat_error`).
extern "C" fn host_pat_cap_kind(i: i64) -> i64 {
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Bytes(..)) => 0,
        Ok(pattern::CaptureValue::Position(_)) => 1,
        Err(e) => {
            pattern::STATE.with(|st| st.borrow_mut().error = e);
            -1
        }
    }
}

extern "C" fn host_pat_cap_str(s: zrtl::StringConstPtr, i: i64) -> StringPtr {
    let src = unsafe { bytes_of(s) };
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Bytes(a, b)) => zrtl::string::string_from_bytes(&src[a..b]),
        _ => zrtl::string::string_from_bytes(b""),
    }
}

extern "C" fn host_pat_cap_pos(i: i64) -> i64 {
    match pattern::capture_text(i as usize) {
        Ok(pattern::CaptureValue::Position(p)) => p as i64,
        _ => 0,
    }
}

extern "C" fn host_pat_error() -> StringPtr {
    pattern::STATE.with(|st| zrtl::string::string_from_bytes(st.borrow().error.as_bytes()))
}

// A stack of byte buffers for `gsub`: a replacement function may run
// a `gsub` of its own.
thread_local! {
    static BUFFERS: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
}

extern "C" fn host_buf_open() {
    BUFFERS.with(|b| b.borrow_mut().push(Vec::new()));
}

extern "C" fn host_buf_push(s: zrtl::StringConstPtr) {
    let bytes = unsafe { bytes_of(s) };
    BUFFERS.with(|b| {
        if let Some(top) = b.borrow_mut().last_mut() {
            top.extend_from_slice(bytes);
        }
    });
}

/// Bytes `a..b` (0-based, `b` excluded) of `s`.
extern "C" fn host_buf_push_range(s: zrtl::StringConstPtr, a: i64, b: i64) {
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
extern "C" fn host_buf_expand(s: zrtl::StringConstPtr, repl: zrtl::StringConstPtr) -> i64 {
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
extern "C" fn host_replace_dots(s: zrtl::StringConstPtr) -> StringPtr {
    let bytes: Vec<u8> = unsafe { bytes_of(s) }
        .iter()
        .map(|&b| if b == b'.' { b'/' } else { b })
        .collect();
    zrtl::string::string_from_bytes(&bytes)
}

extern "C" fn host_buf_close() -> StringPtr {
    let bytes = BUFFERS.with(|b| b.borrow_mut().pop().unwrap_or_default());
    zrtl::string::string_from_bytes(&bytes)
}

// ─── utf8 ───────────────────────────────────────────────────────────
// Lua's own UTF-8: sequences up to six bytes for codes to 0x7FFFFFFF,
// strict by default (no surrogates, nothing past 0x10FFFF).

const MAXUTF: u32 = 0x7FFF_FFFF;
const MAXUNICODE: u32 = 0x10_FFFF;

fn is_cont(b: u8) -> bool {
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
fn u_posrelat(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if (pos.unsigned_abs() as usize) > len {
        0
    } else {
        len as i64 + pos + 1
    }
}

extern "C" fn host_utf8_char(code: i64) -> StringPtr {
    let mut out = Vec::new();
    utf8_encode(code as u32, &mut out);
    zrtl::string::string_from_bytes(&out)
}

/// `utf8.len`: the count, -1 or -2 for a bad initial or final
/// position, or `-(pos) - 2` where decoding fails at 1-based `pos`.
extern "C" fn host_utf8_len(s: zrtl::StringConstPtr, i: i64, j: i64, lax: bool) -> i64 {
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
extern "C" fn host_utf8_offset(s: zrtl::StringConstPtr, n: i64, i: i64, has_i: bool) -> i64 {
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

thread_local! {
    static UTF8_CODES: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `utf8.codepoint`: the codes between `i` and `j` held for
/// `$Lua$utf8_code_at`; their count, or -1 / -2 for a bad position,
/// -3 for an invalid sequence.
extern "C" fn host_utf8_codepoint(
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

extern "C" fn host_utf8_code_at(k: i64) -> i64 {
    UTF8_CODES.with(|c| c.borrow().get(k as usize).copied().unwrap_or(0) as i64)
}

/// `utf8.codes`'s step: from the character at 1-based `n` (0 to
/// start), the next character's position with its code held, 0 at the
/// end, -1 for an invalid sequence.
extern "C" fn host_utf8_next(s: zrtl::StringConstPtr, n: i64, lax: bool) -> i64 {
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

// ─── load ───────────────────────────────────────────────────────────

thread_local! {
    static LOAD_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static LOADS: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// Chunks `load` compiles carry numbers from here, past any file the
/// program was compiled with.
const LOAD_CHUNKS_FROM: i64 = 1 << 20;

/// The chunk name as `luaO_chunkid` spells it: `=name` and `@name` as
/// given, anything else as `[string "..."]` cut at the first line.
fn chunk_name(name: &[u8], source: &[u8]) -> String {
    let text = if name.is_empty() { source } else { name };
    if let Some(rest) = text.strip_prefix(b"=").or_else(|| text.strip_prefix(b"@")) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    const IDSIZE: usize = 60;
    let room = IDSIZE - "[string \"".len() - "...\"]".len() - 1;
    let first_line = text.split(|&b| b == b'\n').next().unwrap_or(b"");
    let cut = first_line.len() < text.len() || first_line.len() > room;
    let shown = if first_line.len() > room {
        &first_line[..room]
    } else {
        first_line
    };
    let shown = String::from_utf8_lossy(shown);
    if cut {
        format!("[string \"{shown}...\"]")
    } else {
        format!("[string \"{shown}\"]")
    }
}

/// `load(source, name)`: the chunk as a function value, or null with
/// the message held for `$Lua$load_error`.
extern "C" fn host_load(
    source: zrtl::StringConstPtr,
    name: zrtl::StringConstPtr,
    env: *const DynamicBox,
) -> *const DynamicBox {
    let (source, name) = unsafe { (bytes_of(source), bytes_of(name)) };
    let chunk_name = chunk_name(name, source);
    let text = crate::source_text(source);
    let index = LOADS.with(|n| {
        let k = n.get();
        n.set(k + 1);
        LOAD_CHUNKS_FROM + k
    });
    match crate::load_chunk(&text, &chunk_name, index, env) {
        Ok(record) => record,
        Err(message) => {
            LOAD_ERROR.with(|e| *e.borrow_mut() = message);
            std::ptr::null()
        }
    }
}

/// `loadfile`/`dofile`: the file's bytes, or null with the message
/// held for `$Lua$load_error`.
extern "C" fn host_read_file(path: zrtl::StringConstPtr) -> StringPtr {
    let path = String::from_utf8_lossy(unsafe { bytes_of(path) }).into_owned();
    match std::fs::read(&path) {
        Ok(bytes) => {
            // A byte order mark is skipped, then a leading `#` line, as
            // `lua` skips a shebang.
            let bytes = bytes
                .strip_prefix(b"\xEF\xBB\xBF")
                .map_or(&bytes[..], |rest| rest)
                .to_vec();
            let bytes = if bytes.first() == Some(&b'#') {
                match bytes.iter().position(|&b| b == b'\n') {
                    Some(nl) => &bytes[nl..],
                    None => &[][..],
                }
            } else {
                &bytes[..]
            };
            zrtl::string::string_from_bytes(bytes)
        }
        Err(e) => {
            let reason = e
                .to_string()
                .split(" (os error")
                .next()
                .unwrap_or("")
                .to_string();
            LOAD_ERROR.with(|err| *err.borrow_mut() = format!("cannot open {path}: {reason}"));
            std::ptr::null_mut()
        }
    }
}

extern "C" fn host_load_error() -> StringPtr {
    LOAD_ERROR.with(|e| zrtl::string::string_from_bytes(e.borrow().as_bytes()))
}

/// `package.searchpath`: the first template of `path` that names a
/// readable file with `name` (its `sep` turned into `rep`) put in
/// for `?`; null when none does, with every file tried listed as
/// the reference lists them for `load_error`.
extern "C" fn host_searchpath(
    name: zrtl::StringConstPtr,
    path: zrtl::StringConstPtr,
    sep: zrtl::StringConstPtr,
    rep: zrtl::StringConstPtr,
) -> StringPtr {
    let (name, path, sep, rep) =
        unsafe { (bytes_of(name), bytes_of(path), bytes_of(sep), bytes_of(rep)) };
    let name = if sep.is_empty() {
        name.to_vec()
    } else {
        replace_all(name, sep, rep)
    };
    // Every piece between separators is a template, the empty ones
    // included: the reference tries and reports them like any other.
    let mut tried = Vec::new();
    for template in path.split(|&b| b == b';') {
        let filename = replace_all(template, b"?", &name);
        let text = String::from_utf8_lossy(&filename).into_owned();
        if std::fs::File::open(&text).is_ok() {
            return zrtl::string::string_from_bytes(&filename);
        }
        tried.push(format!("no file '{text}'"));
    }
    LOAD_ERROR.with(|e| *e.borrow_mut() = tried.join("\n\t"));
    std::ptr::null_mut()
}

fn replace_all(s: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if !from.is_empty() && s[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

// ─── the collector ──────────────────────────────────────────────────

/// `collectgarbage`: 0 runs a collection, 1 answers the bytes the last
/// collection reached.
extern "C" fn host_gc(op: i64) -> i64 {
    match op {
        0 => {
            zyntax_compiler::collector::collect();
            0
        }
        _ => zyntax_compiler::collector::stats().live as i64,
    }
}

/// `os.setlocale`: only the C locale is on offer, so a query or a
/// request for it answers "C" and any other locale is refused.
extern "C" fn host_setlocale(locale: zrtl::StringConstPtr) -> StringPtr {
    let wanted = if locale.is_null() {
        &b""[..]
    } else {
        unsafe { bytes_of(locale) }
    };
    match wanted {
        b"" | b"C" | b"POSIX" => zrtl::string::string_from_bytes(b"C"),
        _ => std::ptr::null_mut(),
    }
}

// ─── uncaught errors ────────────────────────────────────────────────

/// An error nothing caught, on stderr the way `lua` reports it, then
/// the end of the program with status 1. It ends as a program that
/// returns does: every open stream flushed, no destructor run, so a
/// compile thread still running is not raced by the process's teardown.
extern "C" fn host_report_pending(err: *const DynamicBox) {
    use std::io::Write;
    let mut out = b"lua: ".to_vec();
    out.extend(uncaught_text(err));
    out.push(b'\n');
    host_io::host_io_flush_all();
    let _ = std::io::stderr().write_all(&out);
    // SAFETY: `_exit` does not return, and the streams are flushed.
    unsafe { libc::_exit(1) }
}

/// The message of an uncaught error: a string as it is, a number as
/// `tostring` writes it, a table as its `__tostring` gives it when that
/// is a string (or by the error that raises), a file as it prints;
/// anything else by its type.
fn uncaught_text(err: *const DynamicBox) -> Vec<u8> {
    let named = |kind: &str| format!("(error object is a {kind} value)").into_bytes();
    let kind_of = |b: &DynamicBox| b.tag.raw() >> 8;
    let instance = |k: usize| zyntax_builtins::instance_tag(k) as u32 >> 8;
    // SAFETY: the library hands over the pending error, a live box or null.
    let boxed = unsafe { err.as_ref() };
    if boxed.is_some_and(|b| kind_of(b) == instance(super::library::NIL_ERROR_KIND)) {
        return named("nil");
    }
    match unsafe { read_arg(err) } {
        Arg::Nil => named("nil"),
        Arg::Bool(_) => named("boolean"),
        Arg::Int(n) => n.to_string().into_bytes(),
        Arg::Float(x) => float_text(x).into_bytes(),
        Arg::Str(bytes, _) => bytes.to_vec(),
        Arg::Other(text, _) => {
            let b = boxed.expect("a value that is not nil is a box");
            match type_word(b) {
                "FILE*" => text,
                "table" if kind_of(b) == instance(super::library::TABLE_KIND) => {
                    match table_error_text(err) {
                        Ok(Some(text)) => text,
                        Ok(None) => named("table"),
                        Err(why) => {
                            eprintln!("zylua: the error's __tostring could not be compiled: {why}");
                            named("table")
                        }
                    }
                }
                "table" => named("userdata"),
                word => named(word),
            }
        }
    }
}

/// What an uncaught table error's `__tostring` gives, when that is a
/// string, or the message of the error it raises. No chunk reaches the
/// library's text for it, so it is compiled here, once, when the first
/// such error is reported.
fn table_error_text(err: *const DynamicBox) -> Result<Option<Vec<u8>>, String> {
    use crate::lower::{ERROR_TEXT_ENTRY, error_text_program};
    let runtime = crate::runtime().ok_or("no runtime to compile into")?;
    let entry = match runtime.function_pointer(ERROR_TEXT_ENTRY) {
        Some(entry) => entry,
        None => {
            let library = crate::library().map_err(|e| e.to_string())?;
            runtime.declare_entry_points([ERROR_TEXT_ENTRY]);
            runtime
                .compile_typed_program(error_text_program(&library))
                .map_err(|e| e.to_string())?;
            runtime
                .function_pointer(ERROR_TEXT_ENTRY)
                .ok_or("the compiled program has no entry")?
        }
    };
    // SAFETY: the entry was compiled with this signature.
    let text: extern "C" fn(*const DynamicBox) -> *const DynamicBox =
        unsafe { std::mem::transmute(entry) };
    Ok(match unsafe { read_arg(text(err)) } {
        Arg::Str(bytes, _) => Some(bytes.to_vec()),
        _ => None,
    })
}

// ─── the plugin ─────────────────────────────────────────────────────

static INFO: zrtl::ZrtlInfo = zrtl::ZrtlInfo::new(c"lua_host".as_ptr());
static SYMBOLS: [zrtl::ZrtlSymbol; 97] = [
    zrtl::ZrtlSymbol::new(c"$Lua$argc".as_ptr(), host_argc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$argv".as_ptr(), host_argv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$clock".as_ptr(), host_clock as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$time".as_ptr(), host_time as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$float_str".as_ptr(), host_float_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$number_kind".as_ptr(), host_number_kind as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$parse_int".as_ptr(), host_parse_int as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$parse_float".as_ptr(), host_parse_float as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$float_is_int".as_ptr(),
        host_float_is_int as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$format".as_ptr(), host_format as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$format_g".as_ptr(), host_format_g as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$word".as_ptr(), host_word as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$sub".as_ptr(), host_sub as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$byte_at".as_ptr(), host_byte_at as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$from_byte".as_ptr(), host_from_byte as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$reverse".as_ptr(), host_reverse as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$rep".as_ptr(), host_rep as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$find_plain".as_ptr(), host_find_plain as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$bytes".as_ptr(), host_bytes as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$upper".as_ptr(), host_upper as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$lower".as_ptr(), host_lower as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$random_seed".as_ptr(), host_random_seed as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$random_next".as_ptr(), host_random_next as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_seed_now".as_ptr(),
        host_random_seed_now as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_seed_pid".as_ptr(),
        host_random_seed_pid as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$random_float".as_ptr(),
        host_random_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$random_int".as_ptr(), host_random_int as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_specials".as_ptr(),
        host_pat_specials as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_find".as_ptr(), host_pat_find as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_match_at".as_ptr(),
        host_pat_match_at as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_end".as_ptr(), host_pat_end as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_level".as_ptr(), host_pat_level as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$pat_cap_kind".as_ptr(),
        host_pat_cap_kind as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_cap_str".as_ptr(), host_pat_cap_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_cap_pos".as_ptr(), host_pat_cap_pos as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pat_error".as_ptr(), host_pat_error as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_open".as_ptr(), host_buf_open as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_push".as_ptr(), host_buf_push as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$buf_push_range".as_ptr(),
        host_buf_push_range as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_expand".as_ptr(), host_buf_expand as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$buf_close".as_ptr(), host_buf_close as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_char".as_ptr(), host_utf8_char as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_len".as_ptr(), host_utf8_len as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_offset".as_ptr(), host_utf8_offset as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$utf8_codepoint".as_ptr(),
        host_utf8_codepoint as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$utf8_code_at".as_ptr(),
        host_utf8_code_at as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$utf8_next".as_ptr(), host_utf8_next as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$gc".as_ptr(), host_gc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$load".as_ptr(), host_load as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$load_error".as_ptr(), host_load_error as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$report_pending".as_ptr(),
        host_report_pending as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$searchpath".as_ptr(), host_searchpath as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$read_file".as_ptr(), host_read_file as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$replace_dots".as_ptr(),
        host_replace_dots as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$os_error".as_ptr(),
        host_os::host_os_error as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$date_field".as_ptr(),
        host_os::host_date_field as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$time_of".as_ptr(), host_os::host_time_of as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$date_check".as_ptr(),
        host_os::host_date_check as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$date".as_ptr(), host_os::host_date as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$getenv".as_ptr(), host_os::host_getenv as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$tmpname".as_ptr(), host_os::host_tmpname as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$remove".as_ptr(), host_os::host_remove as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$rename".as_ptr(), host_os::host_rename as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$execute".as_ptr(), host_os::host_execute as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$exec_result".as_ptr(),
        host_os::host_exec_result as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$setlocale".as_ptr(), host_setlocale as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pack".as_ptr(), host_pack as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$pack_error".as_ptr(), host_pack_error as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$packsize".as_ptr(), host_packsize as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack".as_ptr(), host_unpack as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_kind".as_ptr(), host_unpack_kind as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_int".as_ptr(), host_unpack_int as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$unpack_float".as_ptr(),
        host_unpack_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_str".as_ptr(), host_unpack_str as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$unpack_next".as_ptr(), host_unpack_next as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_error".as_ptr(),
        host_io::host_io_error as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_reason".as_ptr(),
        host_io::host_io_reason as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_errno".as_ptr(),
        host_io::host_io_errno as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_failed".as_ptr(),
        host_io::host_io_failed as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$io_std".as_ptr(), host_io::host_io_std as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$io_open".as_ptr(), host_io::host_io_open as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_popen".as_ptr(),
        host_io::host_io_popen as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_tmpfile".as_ptr(),
        host_io::host_io_tmpfile as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_is_open".as_ptr(),
        host_io::host_io_is_open as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_is_pipe".as_ptr(),
        host_io::host_io_is_pipe as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_close".as_ptr(),
        host_io::host_io_close as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_line".as_ptr(),
        host_io::host_io_read_line as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_all".as_ptr(),
        host_io::host_io_read_all as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_bytes".as_ptr(),
        host_io::host_io_read_bytes as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_read_number".as_ptr(),
        host_io::host_io_read_number as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_number_int".as_ptr(),
        host_io::host_io_number_int as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_number_float".as_ptr(),
        host_io::host_io_number_float as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_write".as_ptr(),
        host_io::host_io_write as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$io_seek".as_ptr(), host_io::host_io_seek as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_flush".as_ptr(),
        host_io::host_io_flush as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_flush_all".as_ptr(),
        host_io::host_io_flush_all as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$io_setvbuf".as_ptr(),
        host_io::host_io_setvbuf as *const u8,
    ),
];

/// A host symbol the shared library declares for another frontend's
/// host: no Lua value reaches the code that calls it, so reaching it
/// is a fault in the library's dispatch.
extern "C" fn host_unprovided() {
    eprintln!("zylua: the shared library called a host function this frontend does not provide");
    std::process::abort();
}

/// The host's symbols as a plugin the runtime links like any other,
/// with a trap under every host symbol the shared library declares
/// and this host does not define.
pub(crate) fn static_plugin() -> zrtl::StaticPlugin {
    static ALL: OnceLock<&'static [zrtl::ZrtlSymbol]> = OnceLock::new();
    let symbols = ALL.get_or_init(|| {
        let provided: std::collections::HashSet<&str> = SYMBOLS
            .iter()
            .filter_map(|s| {
                // SAFETY: every entry's name is a C string literal.
                unsafe { std::ffi::CStr::from_ptr(s.name) }.to_str().ok()
            })
            .collect();
        let mut all: Vec<zrtl::ZrtlSymbol> = SYMBOLS
            .iter()
            .map(|s| zrtl::ZrtlSymbol::new(s.name, s.ptr))
            .collect();
        for name in crate::fallible::HOST_EXTERNS {
            if provided.contains(name) {
                continue;
            }
            let c: &'static std::ffi::CStr = Box::leak(
                std::ffi::CString::new(*name)
                    .expect("a symbol name")
                    .into_boxed_c_str(),
            );
            all.push(zrtl::ZrtlSymbol::new(
                c.as_ptr(),
                host_unprovided as *const u8,
            ));
        }
        Box::leak(all.into_boxed_slice())
    });
    zrtl::StaticPlugin {
        info: &INFO,
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_lua_prints_them() {
        assert_eq!(float_text(3.0), "3.0");
        assert_eq!(float_text(0.1), "0.1");
        assert_eq!(float_text(1e15), "1e+15");
        assert_eq!(float_text(1e14), "1e+14");
        assert_eq!(float_text(2f64.powi(53)), "9.007199254741e+15");
        assert_eq!(float_text(-0.5), "-0.5");
        assert_eq!(float_text(f64::INFINITY), "inf");
        assert_eq!(float_text(f64::NAN), "nan");
        assert_eq!(float_text(100.0), "100.0");
        assert_eq!(float_text(1.0 / 3.0), "0.33333333333333");
        assert_eq!(float_text(123456.789), "123456.789");
    }

    #[test]
    fn numerals_parse_as_lua_reads_them() {
        assert!(matches!(parse_numeral("42"), Numeral::Int(42)));
        assert!(matches!(parse_numeral("  -7 "), Numeral::Int(-7)));
        assert!(matches!(parse_numeral("0x10"), Numeral::Int(16)));
        assert!(matches!(
            parse_numeral("0xffffffffffffffff"),
            Numeral::Int(-1)
        ));
        assert!(matches!(parse_numeral("1e2"), Numeral::Float(v) if v == 100.0));
        assert!(matches!(parse_numeral("0x1p4"), Numeral::Float(v) if v == 16.0));
        assert!(matches!(parse_numeral(".5"), Numeral::Float(v) if v == 0.5));
        assert!(matches!(parse_numeral("5."), Numeral::Float(v) if v == 5.0));
        assert!(matches!(parse_numeral("abc"), Numeral::None));
        assert!(matches!(parse_numeral(""), Numeral::None));
        assert!(matches!(parse_numeral("1 2"), Numeral::None));
        assert!(matches!(
            parse_numeral("9223372036854775808"),
            Numeral::Float(v) if v == 9223372036854775808.0
        ));
        assert!(matches!(
            parse_numeral("-9223372036854775808"),
            Numeral::Int(i64::MIN)
        ));
    }

    #[test]
    fn format_covers_the_common_conversions() {
        let f = |fmt: &str, args: &[Arg<'_>]| {
            String::from_utf8(format(fmt.as_bytes(), args).unwrap()).unwrap()
        };
        assert_eq!(f("%d items", &[Arg::Int(3)]), "3 items");
        assert_eq!(
            f(
                "%5d|%-5d|%05d",
                &[Arg::Int(42), Arg::Int(42), Arg::Int(-42)]
            ),
            "   42|42   |-0042"
        );
        assert_eq!(f("%.2f", &[Arg::Float(1.23456)]), "1.23");
        assert_eq!(f("%g", &[Arg::Float(100000.0)]), "100000");
        assert_eq!(f("%g", &[Arg::Float(1e20)]), "1e+20");
        assert_eq!(
            f("%x %X %o", &[Arg::Int(255), Arg::Int(255), Arg::Int(8)]),
            "ff FF 10"
        );
        assert_eq!(
            f("%s and %s", &[Arg::Str(b"a", 0), Arg::Float(1.5)]),
            "a and 1.5"
        );
        assert_eq!(f("%q", &[Arg::Str(b"a\"b\n", 0)]), "\"a\\\"b\\\n\"");
        assert_eq!(f("%5.1s|", &[Arg::Str(b"abc", 0)]), "    a|");
        assert_eq!(f("%%", &[]), "%");
        assert_eq!(f("%e", &[Arg::Float(12345.678)]), "1.234568e+04");
        assert_eq!(f("%c%c", &[Arg::Int(72), Arg::Int(105)]), "Hi");
        assert!(format(b"%d", &[Arg::Str(b"x", 0)]).is_err());
        assert!(format(b"%d", &[]).is_err());
    }
}
