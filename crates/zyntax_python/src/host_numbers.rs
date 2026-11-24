//! The host side of the methods of `int` and `float` that no other
//! primitive gives: the bits of an int, an int as bytes and back, and
//! whether a float is whole.

use zrtl::{StringConstPtr, StringPtr, bytes_new, string_as_bytes};

/// `int.bit_length`: the bits of the magnitude.
pub fn bit_length(v: i64) -> i64 {
    64 - i64::from(v.unsigned_abs().leading_zeros())
}

/// `int.bit_count`: the set bits of the magnitude.
pub fn bit_count(v: i64) -> i64 {
    i64::from(v.unsigned_abs().count_ones())
}

/// The `length` bytes of `v`, two's complement when `signed`, the first
/// of them the most significant when `big`. `None` where `v` does not
/// fit, which `to_bytes` raises OverflowError for.
pub fn to_bytes(v: i64, length: i64, big: bool, signed: bool) -> Option<Vec<u8>> {
    let len = usize::try_from(length).ok()?;
    if v < 0 && !signed {
        return None;
    }
    let bits = i128::from(length) * 8;
    let value = i128::from(v);
    let fits = if signed {
        bits > 0
            && value >= -(1i128 << (bits - 1).min(126))
            && (bits > 126 || value < 1i128 << (bits - 1))
    } else {
        value == 0 || (bits > 0 && (bits > 126 || value < 1i128 << bits))
    };
    if !fits {
        return None;
    }
    let mut out: Vec<u8> = (0..len)
        .map(|i| match i {
            0..8 => ((v >> (8 * i)) & 0xff) as u8,
            _ if v < 0 => 0xff,
            _ => 0,
        })
        .collect();
    if big {
        out.reverse();
    }
    Some(out)
}

/// The int `bytes` spell, or `None` where it does not fit 64 bits.
pub fn from_bytes(bytes: &[u8], big: bool, signed: bool) -> Option<i64> {
    let mut little: Vec<u8> = bytes.to_vec();
    if big {
        little.reverse();
    }
    let negative = signed && little.last().is_some_and(|b| b & 0x80 != 0);
    let fill = if negative { 0xff } else { 0 };
    // Bytes past the eighth must only extend the sign or the zeros.
    if little.iter().skip(8).any(|&b| b != fill) {
        return None;
    }
    let mut raw = [fill; 8];
    for (slot, &b) in raw.iter_mut().zip(&little) {
        *slot = b;
    }
    let value = u64::from_le_bytes(raw);
    if negative || value <= i64::MAX as u64 {
        Some(value as i64)
    } else {
        None
    }
}

pub(crate) extern "C" fn host_int_bit_length(v: i64) -> i64 {
    bit_length(v)
}

pub(crate) extern "C" fn host_int_bit_count(v: i64) -> i64 {
    bit_count(v)
}

/// The bytes of `int.to_bytes`, or null where the int does not fit.
pub(crate) extern "C" fn host_int_to_bytes(
    v: i64,
    length: i64,
    big: i64,
    signed: i64,
) -> StringPtr {
    match to_bytes(v, length, big != 0, signed != 0) {
        Some(bytes) => bytes_new(&bytes),
        None => std::ptr::null_mut(),
    }
}

/// Whether `int.from_bytes` can hold the int `bytes` spell.
pub(crate) extern "C" fn host_int_from_bytes_fits(b: StringConstPtr, big: i64, signed: i64) -> i64 {
    i64::from(from_bytes(unsafe { string_as_bytes(b) }, big != 0, signed != 0).is_some())
}

pub(crate) extern "C" fn host_int_from_bytes(b: StringConstPtr, big: i64, signed: i64) -> i64 {
    from_bytes(unsafe { string_as_bytes(b) }, big != 0, signed != 0).unwrap_or(0)
}

/// Whether the float is finite and whole.
pub(crate) extern "C" fn host_float_is_integer(v: f64) -> i64 {
    i64::from(v.is_finite() && v == v.trunc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_of_the_magnitude() {
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(255), 8);
        assert_eq!(bit_length(256), 9);
        assert_eq!(bit_length(-5), 3);
        assert_eq!(bit_length(i64::MIN), 64);
        assert_eq!(bit_count(255), 8);
        assert_eq!(bit_count(-7), 3);
    }

    #[test]
    fn ints_become_bytes() {
        assert_eq!(to_bytes(258, 2, true, false), Some(vec![1, 2]));
        assert_eq!(to_bytes(258, 2, false, false), Some(vec![2, 1]));
        assert_eq!(to_bytes(256, 1, true, false), None);
        assert_eq!(to_bytes(-1, 2, true, false), None);
        assert_eq!(to_bytes(-1, 2, true, true), Some(vec![0xff, 0xff]));
        assert_eq!(to_bytes(-129, 1, true, true), None);
        assert_eq!(to_bytes(127, 1, true, true), Some(vec![127]));
        assert_eq!(to_bytes(128, 1, true, true), None);
        assert_eq!(to_bytes(0, 0, true, false), Some(vec![]));
        assert_eq!(to_bytes(1, 0, true, false), None);
        assert_eq!(
            to_bytes(-2, 10, false, true),
            Some(vec![
                0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
            ])
        );
    }

    #[test]
    fn bytes_become_ints() {
        assert_eq!(from_bytes(&[1, 2], true, false), Some(258));
        assert_eq!(from_bytes(&[1, 2], false, false), Some(513));
        assert_eq!(from_bytes(&[0xff, 0xff], true, true), Some(-1));
        assert_eq!(from_bytes(&[0xff, 0xff], true, false), Some(65535));
        assert_eq!(from_bytes(&[], true, false), Some(0));
        assert_eq!(from_bytes(&[0x80; 9], true, false), None);
        assert_eq!(from_bytes(&[0xff; 9], true, true), Some(-1));
        assert_eq!(from_bytes(&[0xff; 8], true, false), None);
    }
}
