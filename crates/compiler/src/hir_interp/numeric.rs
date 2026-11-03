//! Scalar reads used by arithmetic and comparison bytecodes.

use super::{InterpError, ZyntaxValue, value_to_i64};

#[inline]
pub(super) fn ireg_i64(value: &ZyntaxValue) -> Result<i64, InterpError> {
    if let ZyntaxValue::Int(value) = value {
        return Ok(*value);
    }
    other_integer(value)
}

// Keep general coercion and diagnostic construction out of the common Int path.
#[inline(never)]
fn other_integer(value: &ZyntaxValue) -> Result<i64, InterpError> {
    value_to_i64(value).ok_or_else(|| InterpError::TypeMismatch {
        expected: "integer".to_string(),
        got: format!("{:?}", value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_reads_preserve_width_sign_and_coercions() {
        let cases = [
            (ZyntaxValue::Int(i64::MIN), i64::MIN),
            (ZyntaxValue::Int(i64::MAX), i64::MAX),
            (ZyntaxValue::Int(0), 0),
            (ZyntaxValue::UInt(u64::MAX), -1),
            (ZyntaxValue::I8(-128), -128),
            (ZyntaxValue::I16(-32768), -32768),
            (ZyntaxValue::I32(i32::MIN), i32::MIN as i64),
            (ZyntaxValue::U8(u8::MAX), u8::MAX as i64),
            (ZyntaxValue::U16(u16::MAX), u16::MAX as i64),
            (ZyntaxValue::U32(u32::MAX), u32::MAX as i64),
            (ZyntaxValue::Bool(true), 1),
            (ZyntaxValue::Bool(false), 0),
            (ZyntaxValue::Float(-3.75), -3),
            (ZyntaxValue::Float(f64::NAN), 0),
            (ZyntaxValue::Float(f64::INFINITY), i64::MAX),
            (ZyntaxValue::F32(3.75), 3),
            (
                ZyntaxValue::Pointer(std::ptr::without_provenance_mut(0x1234)),
                0x1234,
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(ireg_i64(&value).unwrap(), expected, "{value:?}");
        }
    }

    #[test]
    fn nonnumeric_values_keep_type_mismatch_diagnostics() {
        for value in [
            ZyntaxValue::Undef,
            ZyntaxValue::Null,
            ZyntaxValue::String("no".into()),
        ] {
            let Err(InterpError::TypeMismatch { expected, got }) = ireg_i64(&value) else {
                panic!("expected an integer type error for {value:?}");
            };
            assert_eq!(expected, "integer");
            assert_eq!(got, format!("{value:?}"));
        }
    }
}
