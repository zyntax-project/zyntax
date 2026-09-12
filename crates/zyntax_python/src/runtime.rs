//! The natives a compiled Python program links against.
//!
//! What the language defines and the IR does not: how a value prints,
//! and the builtins that convert between the primitive types. Each takes
//! a `DynamicBox` and reads its type tag, since a Python value's type is
//! a runtime fact. [`plugin`] hands them to a runtime as one statically
//! registered symbol table; [`super::BUILTINS`] is the frontend's view
//! of the same list.

use std::fmt::Write as _;
use std::io::Write as _;
use zrtl::{
    string_data, string_length, string_new, zrtl_symbol_sig, DynamicBox, StaticPlugin, StringPtr,
    TypeCategory, ZrtlInfo, ZrtlSymbol,
};

/// The string a box holds, when it holds one.
unsafe fn string_of(value: &DynamicBox) -> Option<&str> {
    if !value.is_category(TypeCategory::String) || value.data.is_null() {
        return None;
    }
    let ptr = value.data as zrtl::StringConstPtr;
    let len = string_length(ptr) as usize;
    let data = string_data(ptr);
    if len == 0 || data.is_null() {
        return Some("");
    }
    std::str::from_utf8(std::slice::from_raw_parts(data, len)).ok()
}

unsafe fn int_of(value: &DynamicBox) -> Option<i64> {
    match value.category() {
        TypeCategory::Int => match value.size {
            1 => value.as_ref::<i8>().map(|v| *v as i64),
            2 => value.as_ref::<i16>().map(|v| *v as i64),
            4 => value.as_ref::<i32>().map(|v| *v as i64),
            _ => value.as_ref::<i64>().copied(),
        },
        TypeCategory::UInt => match value.size {
            1 => value.as_ref::<u8>().map(|v| *v as i64),
            2 => value.as_ref::<u16>().map(|v| *v as i64),
            4 => value.as_ref::<u32>().map(|v| *v as i64),
            _ => value.as_ref::<u64>().map(|v| *v as i64),
        },
        _ => None,
    }
}

unsafe fn bool_of(value: &DynamicBox) -> Option<bool> {
    if !value.is_category(TypeCategory::Bool) {
        return None;
    }
    value.as_ref::<u8>().map(|b| *b != 0)
}

unsafe fn float_of(value: &DynamicBox) -> Option<f64> {
    if !value.is_category(TypeCategory::Float) {
        return None;
    }
    match value.size {
        4 => value.as_ref::<f32>().map(|v| *v as f64),
        _ => value.as_ref::<f64>().copied(),
    }
}

/// `repr(float)`: the shortest digits that round-trip, positional when
/// the exponent is in -4..16 and scientific otherwise, and always with
/// a fractional part or an exponent so it reads as a float.
pub fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    // `{:e}` is the shortest round-trip mantissa with a decimal exponent.
    let sci = format!("{:e}", v);
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut out = String::from(sign);
    if (-4..16).contains(&exp) {
        // The point sits after `exp + 1` digits.
        let point = exp + 1;
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let point = point as usize;
            if digits.len() <= point {
                out.push_str(&digits);
                for _ in digits.len()..point {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// `str(value)`.
unsafe fn str_of(value: &DynamicBox, out: &mut String) {
    match value.category() {
        TypeCategory::Void => out.push_str("None"),
        TypeCategory::Bool => match bool_of(value) {
            Some(true) => out.push_str("True"),
            _ => out.push_str("False"),
        },
        TypeCategory::Int | TypeCategory::UInt => {
            if let Some(i) = int_of(value) {
                let _ = write!(out, "{i}");
            }
        }
        TypeCategory::Float => {
            if let Some(f) = float_of(value) {
                out.push_str(&float_repr(f));
            }
        }
        TypeCategory::String => {
            if let Some(s) = string_of(value) {
                out.push_str(s);
            }
        }
        other => {
            let _ = write!(out, "<{other:?} object at {:p}>", value.data);
        }
    }
}

/// Truthiness: `None`, `False`, zero and the empty string are false.
unsafe fn truthy(value: &DynamicBox) -> bool {
    match value.category() {
        TypeCategory::Void => false,
        TypeCategory::Bool => bool_of(value).unwrap_or(false),
        TypeCategory::Int | TypeCategory::UInt => int_of(value).unwrap_or(0) != 0,
        TypeCategory::Float => float_of(value).unwrap_or(0.0) != 0.0,
        TypeCategory::String => string_of(value).is_some_and(|s| !s.is_empty()),
        _ => !value.data.is_null(),
    }
}

/// `int(s)`: optional sign and decimal digits, whitespace around and
/// underscores between digits allowed. `None` when the text is not an
/// integer literal.
fn parse_int(text: &str) -> Option<i64> {
    let cleaned: String = text.trim().chars().filter(|c| *c != '_').collect();
    cleaned.parse().ok()
}

fn parse_float(text: &str) -> Option<f64> {
    let cleaned: String = text.trim().chars().filter(|c| *c != '_').collect();
    match cleaned.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => cleaned.parse().ok(),
    }
}

/// `print(value, end="")`.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_print(value: *const DynamicBox) {
    let mut out = String::new();
    if value.is_null() {
        out.push_str("None");
    } else {
        str_of(&*value, &mut out);
    }
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
    let _ = lock.flush();
}

/// `print(value)`.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_println(value: *const DynamicBox) {
    let mut out = String::new();
    if value.is_null() {
        out.push_str("None");
    } else {
        str_of(&*value, &mut out);
    }
    out.push('\n');
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
    let _ = lock.flush();
}

/// `str(value)`, as a new ZRTL string.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_str(value: *const DynamicBox) -> StringPtr {
    let mut out = String::new();
    if value.is_null() {
        out.push_str("None");
    } else {
        str_of(&*value, &mut out);
    }
    string_new(&out)
}

/// `int(value)`: a bool or int as itself, a float truncated toward
/// zero, a string parsed. Anything else, or text that is not an
/// integer, is 0.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_int(value: *const DynamicBox) -> i64 {
    if value.is_null() {
        return 0;
    }
    let v = &*value;
    match v.category() {
        TypeCategory::Bool => bool_of(v).unwrap_or(false) as i64,
        TypeCategory::Int | TypeCategory::UInt => int_of(v).unwrap_or(0),
        TypeCategory::Float => float_of(v).unwrap_or(0.0).trunc() as i64,
        TypeCategory::String => string_of(v).and_then(parse_int).unwrap_or(0),
        _ => 0,
    }
}

/// `float(value)`.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_float(value: *const DynamicBox) -> f64 {
    if value.is_null() {
        return 0.0;
    }
    let v = &*value;
    match v.category() {
        TypeCategory::Bool => bool_of(v).unwrap_or(false) as i64 as f64,
        TypeCategory::Int | TypeCategory::UInt => int_of(v).unwrap_or(0) as f64,
        TypeCategory::Float => float_of(v).unwrap_or(0.0),
        TypeCategory::String => string_of(v).and_then(parse_float).unwrap_or(0.0),
        _ => 0.0,
    }
}

/// `bool(value)`.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_bool(value: *const DynamicBox) -> bool {
    !value.is_null() && truthy(&*value)
}

/// `len(value)` of a string, in code points.
///
/// # Safety
/// `value` is null or points to a valid `DynamicBox`.
unsafe extern "C" fn py_len(value: *const DynamicBox) -> i64 {
    if value.is_null() {
        return 0;
    }
    string_of(&*value)
        .map(|s| s.chars().count() as i64)
        .unwrap_or(0)
}

static INFO: ZrtlInfo = ZrtlInfo::new(c"python".as_ptr());

static SYMBOLS: [ZrtlSymbol; 7] = [
    zrtl_symbol_sig!("$Py$print", py_print, dynamic(1) -> void),
    zrtl_symbol_sig!("$Py$println", py_println, dynamic(1) -> void),
    zrtl_symbol_sig!("$Py$str", py_str, dynamic(1) -> dynamic),
    zrtl_symbol_sig!("$Py$int", py_int, dynamic(1) -> i64),
    zrtl_symbol_sig!("$Py$float", py_float, dynamic(1) -> f64),
    zrtl_symbol_sig!("$Py$bool", py_bool, dynamic(1) -> bool),
    zrtl_symbol_sig!("$Py$len", py_len, dynamic(1) -> i64),
];

/// The natives as a plugin a runtime registers without loading anything.
pub fn plugin() -> StaticPlugin {
    StaticPlugin {
        info: &INFO,
        symbols: &SYMBOLS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_repr() {
        for (v, want) in [
            (1.0, "1.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456789.0, "123456789.0"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.5e-7, "2.5e-07"),
            (1.5e300, "1.5e+300"),
            (-2.5, "-2.5"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(float_repr(v), want, "repr({v:?})");
        }
    }

    #[test]
    fn int_parses_like_python() {
        assert_eq!(parse_int(" 42 "), Some(42));
        assert_eq!(parse_int("-7"), Some(-7));
        assert_eq!(parse_int("1_000"), Some(1000));
        assert_eq!(parse_int("4.5"), None);
        assert_eq!(parse_int("abc"), None);
    }
}
