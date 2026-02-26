//! Conversion traits for Zyntax/Rust value interop
//!
//! These traits provide ergonomic bidirectional conversion between
//! Rust types and Zyntax runtime values.

use crate::error::{ConversionError, ConversionResult};
use crate::value::ZyntaxValue;
use zyntax_compiler::zrtl::{DynamicValue, TypeCategory, TypeId};

/// Convert a Zyntax value to a Rust type.
///
/// This trait is implemented for Rust types that can be constructed
/// from Zyntax runtime values. The conversion may fail if the Zyntax
/// value's type doesn't match the expected Rust type.
///
/// # Example
///
/// ```rust,ignore
/// use zyntax_embed::{FromZyntax, ZyntaxValue};
///
/// let zyntax_val = ZyntaxValue::Int(42);
/// let rust_val: i32 = i32::from_zyntax(zyntax_val)?;
/// assert_eq!(rust_val, 42);
/// ```
pub trait FromZyntax: Sized {
    /// Convert from a ZyntaxValue to Self
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self>;

    /// Convert from a DynamicValue to Self
    ///
    /// # Safety
    /// The DynamicValue must have a valid type_meta pointer
    unsafe fn from_dynamic(value: DynamicValue) -> ConversionResult<Self> {
        Self::from_zyntax(ZyntaxValue::from_dynamic(value)?)
    }
}

/// Convert a Rust type to a Zyntax value.
///
/// This trait is implemented for Rust types that can be converted
/// to Zyntax runtime values. The conversion should not fail for
/// well-formed values.
///
/// # Example
///
/// ```rust,ignore
/// use zyntax_embed::{IntoZyntax, ZyntaxValue};
///
/// let rust_val: i32 = 42;
/// let zyntax_val = rust_val.into_zyntax();
/// assert!(matches!(zyntax_val, ZyntaxValue::Int(42)));
/// ```
pub trait IntoZyntax {
    /// Convert Self to a ZyntaxValue
    fn into_zyntax(self) -> ZyntaxValue;

    /// Convert Self to a DynamicValue (heap-allocated)
    fn into_dynamic(self) -> DynamicValue
    where
        Self: Sized,
    {
        self.into_zyntax().into_dynamic()
    }
}

/// Fallible conversion from Zyntax value (for types that may fail conversion)
///
/// Use this trait when conversion might fail due to value constraints,
/// not just type mismatches (e.g., converting to NonZeroI32).
pub trait TryFromZyntax: Sized {
    type Error;

    /// Try to convert from a ZyntaxValue
    fn try_from_zyntax(value: ZyntaxValue) -> Result<Self, Self::Error>;
}

/// Fallible conversion to Zyntax value (for types that may fail conversion)
pub trait TryIntoZyntax {
    type Error;

    /// Try to convert to a ZyntaxValue
    fn try_into_zyntax(self) -> Result<ZyntaxValue, Self::Error>;
}

// ============================================================================
// Primitive Type Implementations
// ============================================================================

impl FromZyntax for i8 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= i8::MIN as i64 && v <= i8::MAX as i64 {
                    Ok(v as i8)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Int,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for i8 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Int(self as i64)
    }
}

impl FromZyntax for i16 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= i16::MIN as i64 && v <= i16::MAX as i64 {
                    Ok(v as i16)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Int,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for i16 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Int(self as i64)
    }
}

impl FromZyntax for i32 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                    Ok(v as i32)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Int,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for i32 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Int(self as i64)
    }
}

impl FromZyntax for i64 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => Ok(v),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Int,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for i64 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Int(self)
    }
}

impl FromZyntax for u8 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= 0 && v <= u8::MAX as i64 {
                    Ok(v as u8)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            ZyntaxValue::UInt(v) => {
                if v <= u8::MAX as u64 {
                    Ok(v as u8)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::UInt,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for u8 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::UInt(self as u64)
    }
}

impl FromZyntax for u16 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= 0 && v <= u16::MAX as i64 {
                    Ok(v as u16)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            ZyntaxValue::UInt(v) => {
                if v <= u16::MAX as u64 {
                    Ok(v as u16)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::UInt,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for u16 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::UInt(self as u64)
    }
}

impl FromZyntax for u32 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= 0 && v <= u32::MAX as i64 {
                    Ok(v as u32)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            ZyntaxValue::UInt(v) => {
                if v <= u32::MAX as u64 {
                    Ok(v as u32)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::UInt,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for u32 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::UInt(self as u64)
    }
}

impl FromZyntax for u64 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Int(v) => {
                if v >= 0 {
                    Ok(v as u64)
                } else {
                    Err(ConversionError::IntegerOverflow { value: v as i128 })
                }
            }
            ZyntaxValue::UInt(v) => Ok(v),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::UInt,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for u64 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::UInt(self)
    }
}

impl FromZyntax for f32 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Float(v) => Ok(v as f32),
            ZyntaxValue::Int(v) => Ok(v as f32),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Float,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for f32 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Float(self as f64)
    }
}

impl FromZyntax for f64 {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Float(v) => Ok(v),
            ZyntaxValue::Int(v) => Ok(v as f64),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Float,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for f64 {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Float(self)
    }
}

impl FromZyntax for bool {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Bool(v) => Ok(v),
            ZyntaxValue::Int(v) => Ok(v != 0),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Bool,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for bool {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Bool(self)
    }
}

impl FromZyntax for String {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::String(s) => Ok(s),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::String,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for String {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::String(self)
    }
}

impl IntoZyntax for &str {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::String(self.to_string())
    }
}

// ============================================================================
// Option<T> Implementation
// ============================================================================

impl<T: FromZyntax> FromZyntax for Option<T> {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Null => Ok(None),
            ZyntaxValue::Optional(inner) => match *inner {
                Some(v) => Ok(Some(T::from_zyntax(v)?)),
                None => Ok(None),
            },
            other => Ok(Some(T::from_zyntax(other)?)),
        }
    }
}

impl<T: IntoZyntax> IntoZyntax for Option<T> {
    fn into_zyntax(self) -> ZyntaxValue {
        match self {
            Some(v) => ZyntaxValue::Optional(Box::new(Some(v.into_zyntax()))),
            None => ZyntaxValue::Optional(Box::new(None)),
        }
    }
}

// ============================================================================
// Vec<T> Implementation
// ============================================================================

impl<T: FromZyntax> FromZyntax for Vec<T> {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Array(arr) => arr
                .into_iter()
                .enumerate()
                .map(|(i, v)| T::from_zyntax(v).map_err(|e| ConversionError::array_element(i, e)))
                .collect(),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Array,
                value.type_category(),
            )),
        }
    }
}

impl<T: IntoZyntax> IntoZyntax for Vec<T> {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Array(self.into_iter().map(|v| v.into_zyntax()).collect())
    }
}

// ============================================================================
// Result<T, E> Implementation
// ============================================================================

impl<T: FromZyntax, E: FromZyntax> FromZyntax for Result<T, E> {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Result(inner) => match *inner {
                Ok(v) => Ok(Ok(T::from_zyntax(v)?)),
                Err(e) => Ok(Err(E::from_zyntax(e)?)),
            },
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Result,
                value.type_category(),
            )),
        }
    }
}

impl<T: IntoZyntax, E: IntoZyntax> IntoZyntax for Result<T, E> {
    fn into_zyntax(self) -> ZyntaxValue {
        match self {
            Ok(v) => ZyntaxValue::Result(Box::new(Ok(v.into_zyntax()))),
            Err(e) => ZyntaxValue::Result(Box::new(Err(e.into_zyntax()))),
        }
    }
}

// ============================================================================
// Unit Type Implementation
// ============================================================================

impl FromZyntax for () {
    fn from_zyntax(value: ZyntaxValue) -> ConversionResult<Self> {
        match value {
            ZyntaxValue::Void => Ok(()),
            ZyntaxValue::Null => Ok(()),
            _ => Err(ConversionError::type_mismatch(
                TypeCategory::Void,
                value.type_category(),
            )),
        }
    }
}

impl IntoZyntax for () {
    fn into_zyntax(self) -> ZyntaxValue {
        ZyntaxValue::Void
    }
}

// ============================================================================
// Blanket Implementations
// ============================================================================

/// Blanket impl: any FromZyntax type also implements TryFromZyntax
impl<T: FromZyntax> TryFromZyntax for T {
    type Error = ConversionError;

    fn try_from_zyntax(value: ZyntaxValue) -> Result<Self, Self::Error> {
        T::from_zyntax(value)
    }
}

/// Blanket impl: any IntoZyntax type also implements TryIntoZyntax (infallible)
impl<T: IntoZyntax> TryIntoZyntax for T {
    type Error = std::convert::Infallible;

    fn try_into_zyntax(self) -> Result<ZyntaxValue, Self::Error> {
        Ok(self.into_zyntax())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i32_roundtrip() {
        let original: i32 = 42;
        let zyntax = original.into_zyntax();
        let back: i32 = i32::from_zyntax(zyntax).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn test_string_roundtrip() {
        let original = "Hello, Zyntax!".to_string();
        let zyntax = original.clone().into_zyntax();
        let back: String = String::from_zyntax(zyntax).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn test_vec_roundtrip() {
        let original: Vec<i32> = vec![1, 2, 3, 4, 5];
        let zyntax = original.clone().into_zyntax();
        let back: Vec<i32> = Vec::from_zyntax(zyntax).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn test_option_roundtrip() {
        let some_val: Option<i32> = Some(42);
        let zyntax = some_val.into_zyntax();
        let back: Option<i32> = Option::from_zyntax(zyntax).unwrap();
        assert_eq!(back, Some(42));

        let none_val: Option<i32> = None;
        let zyntax = none_val.into_zyntax();
        let back: Option<i32> = Option::from_zyntax(zyntax).unwrap();
        assert_eq!(back, None);
    }

    #[test]
    fn test_overflow_error() {
        let big_val = ZyntaxValue::Int(i64::MAX);
        let result: ConversionResult<i32> = i32::from_zyntax(big_val);
        assert!(matches!(
            result,
            Err(ConversionError::IntegerOverflow { .. })
        ));
    }

    #[test]
    fn test_type_mismatch_error() {
        let string_val = ZyntaxValue::String("hello".to_string());
        let result: ConversionResult<i32> = i32::from_zyntax(string_val);
        assert!(matches!(result, Err(ConversionError::TypeMismatch { .. })));
    }
}
