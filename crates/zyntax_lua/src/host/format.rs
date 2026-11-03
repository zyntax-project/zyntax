//! Format support.

use super::*;

pub(super) unsafe fn read_arg<'a>(b: *const DynamicBox) -> Arg<'a> {
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

pub(super) fn type_word(b: &DynamicBox) -> &'static str {
    let kind = b.tag.raw() >> 8;
    if kind == (zyntax_builtins::FUNC_TAG >> 8) as u32 {
        "function"
    } else if kind == zyntax_builtins::instance_tag(crate::library::THREAD_KIND) as u32 >> 8 {
        "thread"
    } else if kind == zyntax_builtins::instance_tag(crate::library::FILE_KIND) as u32 >> 8 {
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

pub(super) fn signed(digits: String, negative: bool, plus: bool, space: bool) -> String {
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

pub(super) fn int_arg(a: &Arg<'_>, n: usize) -> Result<i64, String> {
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

pub(super) fn float_arg(a: &Arg<'_>, n: usize) -> Result<f64, String> {
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

pub(super) fn arg_type(a: &Arg<'_>) -> &'static str {
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
pub(super) fn spec_is_valid(spec: &[u8], allowed: &[u8], precision: bool) -> bool {
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
pub(super) fn quoted(bytes: &[u8]) -> Vec<u8> {
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
pub(super) fn hex_float(v: f64, upper: bool) -> String {
    hex_float_digits(v, upper, None)
}

/// `%.Na`: the mantissa rounded to `precision` hexadecimal digits,
/// to the nearest with a tie going toward zero, as the C library
/// here rounds it; without one, exactly, with no trailing zeros.
pub(super) fn hex_float_digits(v: f64, upper: bool, precision: Option<usize>) -> String {
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

pub(super) unsafe fn args_of<'a>(list: *const ListHeader) -> Vec<Arg<'a>> {
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
pub(super) unsafe fn args_with_texts<'a>(
    list: *const ListHeader,
    texts: *const ListHeader,
) -> Vec<Arg<'a>> {
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
pub(super) extern "C" fn host_format(
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
