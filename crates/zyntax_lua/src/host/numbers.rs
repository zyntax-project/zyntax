//! Numbers support.

use super::*;

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
pub(super) fn format_g_flags(x: f64, p: usize, alt: bool, upper: bool) -> String {
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
pub(super) fn strip_zeros(s: &str) -> String {
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

pub(super) extern "C" fn host_float_str(x: f64) -> StringPtr {
    zrtl::string_new(&float_text(x))
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

pub(super) fn parse_decimal(s: &str, negative: bool) -> Numeral {
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

pub(super) fn parse_hex(s: &str) -> Numeral {
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
pub(super) fn ldexp(mut r: f64, mut e: i64) -> f64 {
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

pub(super) unsafe fn text_of(s: zrtl::StringConstPtr) -> &'static str {
    unsafe { zrtl::string_as_str(s) }.unwrap_or("")
}

/// 0 when the text is no numeral, 1 for an integer, 2 for a float.
pub(super) extern "C" fn host_number_kind(s: zrtl::StringConstPtr) -> i64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::None => 0,
        Numeral::Int(_) => 1,
        Numeral::Float(_) => 2,
    }
}

pub(super) extern "C" fn host_parse_int(s: zrtl::StringConstPtr) -> i64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::Int(v) => v,
        _ => 0,
    }
}

pub(super) extern "C" fn host_parse_float(s: zrtl::StringConstPtr) -> f64 {
    match parse_numeral(unsafe { text_of(s) }) {
        Numeral::Float(v) => v,
        Numeral::Int(v) => v as f64,
        Numeral::None => 0.0,
    }
}

/// Whether a float has an exact integer value that fits.
pub(super) extern "C" fn host_float_is_int(x: f64) -> bool {
    x.fract() == 0.0 && (-9223372036854775808.0..9223372036854775808.0).contains(&x)
}
