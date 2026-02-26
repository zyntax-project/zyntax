//! ZyntaxValue - Runtime value representation for Rust interop
//!
//! This module provides `ZyntaxValue`, a Rust-friendly enum that represents
//! any Zyntax runtime value. It serves as the intermediate representation
//! for converting between Rust types and Zyntax's `DynamicValue`.

use crate::error::{ConversionError, ConversionResult};
use std::collections::HashMap;
use zyntax_compiler::zrtl::{DynamicValue, TypeCategory, TypeId, TypeMeta};

/// A Rust-friendly representation of a Zyntax runtime value.
///
/// This enum provides a safe, owned representation of Zyntax values that can be
/// easily pattern-matched and converted to/from native Rust types.
///
/// # Memory Safety
///
/// Unlike `DynamicValue` which uses raw pointers, `ZyntaxValue` is fully owned
/// and follows Rust's standard ownership semantics. Converting between the two
/// requires explicit memory management.
#[derive(Debug, Clone, PartialEq)]
pub enum ZyntaxValue {
    /// Void/unit type (no value)
    Void,

    /// Null value
    Null,

    /// Boolean value
    Bool(bool),

    /// Signed integer (stored as i64 for flexibility)
    Int(i64),

    /// Unsigned integer (stored as u64 for flexibility)
    UInt(u64),

    /// Floating point (stored as f64 for precision)
    Float(f64),

    /// String value (owned, UTF-8)
    String(String),

    /// Array of values
    Array(Vec<ZyntaxValue>),

    /// Map/Dictionary (String keys for simplicity)
    Map(HashMap<String, ZyntaxValue>),

    /// Struct with named fields
    Struct {
        /// Type name of the struct
        type_name: String,
        /// Field values indexed by field name
        fields: HashMap<String, ZyntaxValue>,
    },

    /// Enum variant
    Enum {
        /// Type name of the enum
        type_name: String,
        /// Variant name
        variant: String,
        /// Associated data (if any)
        data: Option<Box<ZyntaxValue>>,
    },

    /// Optional value (Some or None)
    Optional(Box<Option<ZyntaxValue>>),

    /// Result value (Ok or Err)
    Result(Box<Result<ZyntaxValue, ZyntaxValue>>),

    /// Tuple of values
    Tuple(Vec<ZyntaxValue>),

    /// Function reference (stored as opaque pointer)
    Function {
        /// Function pointer (as usize for portability)
        ptr: usize,
        /// Optional function name
        name: Option<String>,
    },

    /// Raw pointer (for FFI interop)
    Pointer(*mut u8),

    /// Opaque/Dynamic value (for types we can't introspect)
    ///
    /// This variant preserves full type metadata from the Zyntax runtime,
    /// enabling proper memory management (via the dropper function) and
    /// type introspection (via generic args).
    ///
    /// # Memory Ownership
    ///
    /// When `owned` is true, the value will be dropped using the TypeMeta's
    /// dropper function when this ZyntaxValue is dropped. When false, the
    /// caller is responsible for memory management.
    Opaque {
        /// Pointer to type metadata (contains dropper, generic args, etc.)
        /// This may point to static metadata or heap-allocated metadata.
        type_meta: *const TypeMeta,
        /// Raw pointer to the actual value data
        ptr: *mut u8,
        /// Whether we own this memory (should drop on destruction)
        owned: bool,
    },
}

// ZyntaxValue contains raw pointers in some variants, but we manage them carefully
// For Pointer and Opaque variants, the caller is responsible for memory safety
unsafe impl Send for ZyntaxValue {}
unsafe impl Sync for ZyntaxValue {}

impl ZyntaxValue {
    /// Get the type category of this value
    pub fn type_category(&self) -> TypeCategory {
        match self {
            ZyntaxValue::Void => TypeCategory::Void,
            ZyntaxValue::Null => TypeCategory::Void,
            ZyntaxValue::Bool(_) => TypeCategory::Bool,
            ZyntaxValue::Int(_) => TypeCategory::Int,
            ZyntaxValue::UInt(_) => TypeCategory::UInt,
            ZyntaxValue::Float(_) => TypeCategory::Float,
            ZyntaxValue::String(_) => TypeCategory::String,
            ZyntaxValue::Array(_) => TypeCategory::Array,
            ZyntaxValue::Map(_) => TypeCategory::Map,
            ZyntaxValue::Struct { .. } => TypeCategory::Struct,
            ZyntaxValue::Enum { .. } => TypeCategory::Enum,
            ZyntaxValue::Optional(_) => TypeCategory::Optional,
            ZyntaxValue::Result(_) => TypeCategory::Result,
            ZyntaxValue::Tuple(_) => TypeCategory::Tuple,
            ZyntaxValue::Function { .. } => TypeCategory::Function,
            ZyntaxValue::Pointer(_) => TypeCategory::Pointer,
            ZyntaxValue::Opaque { .. } => TypeCategory::Opaque,
        }
    }

    /// Check if this is a null or void value
    pub fn is_null_or_void(&self) -> bool {
        matches!(self, ZyntaxValue::Void | ZyntaxValue::Null)
    }

    /// Check if this is an integer (signed or unsigned)
    pub fn is_integer(&self) -> bool {
        matches!(self, ZyntaxValue::Int(_) | ZyntaxValue::UInt(_))
    }

    /// Check if this is numeric (integer or float)
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            ZyntaxValue::Int(_) | ZyntaxValue::UInt(_) | ZyntaxValue::Float(_)
        )
    }

    /// Try to get as a signed integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ZyntaxValue::Int(v) => Some(*v),
            ZyntaxValue::UInt(v) if *v <= i64::MAX as u64 => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to get as an i32 (convenience method)
    pub fn as_i32(&self) -> Option<i32> {
        self.as_int().and_then(|v| {
            if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                Some(v as i32)
            } else {
                None
            }
        })
    }

    /// Try to get as an i64 (convenience alias for as_int)
    pub fn as_i64(&self) -> Option<i64> {
        self.as_int()
    }

    /// Try to get as an unsigned integer
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            ZyntaxValue::UInt(v) => Some(*v),
            ZyntaxValue::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to get as a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ZyntaxValue::Float(v) => Some(*v),
            ZyntaxValue::Int(v) => Some(*v as f64),
            ZyntaxValue::UInt(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to get as a string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ZyntaxValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to get as a bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ZyntaxValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as an array reference
    pub fn as_array(&self) -> Option<&[ZyntaxValue]> {
        match self {
            ZyntaxValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    /// Try to get a struct field by name
    pub fn get_field(&self, name: &str) -> Option<&ZyntaxValue> {
        match self {
            ZyntaxValue::Struct { fields, .. } => fields.get(name),
            ZyntaxValue::Map(map) => map.get(name),
            _ => None,
        }
    }

    /// Check if this is an opaque value
    pub fn is_opaque(&self) -> bool {
        matches!(self, ZyntaxValue::Opaque { .. })
    }

    /// Get the TypeMeta for an opaque value
    ///
    /// Returns `None` if this is not an `Opaque` variant or if the type_meta is null.
    pub fn opaque_type_meta(&self) -> Option<&TypeMeta> {
        match self {
            ZyntaxValue::Opaque { type_meta, .. } if !type_meta.is_null() => unsafe {
                Some(&**type_meta)
            },
            _ => None,
        }
    }

    /// Get the TypeId from an opaque value's TypeMeta
    ///
    /// Returns `None` if this is not an `Opaque` variant or if the type_meta is null.
    pub fn opaque_type_id(&self) -> Option<TypeId> {
        self.opaque_type_meta().map(|meta| meta.type_id)
    }

    /// Get the raw pointer from an opaque value
    pub fn opaque_ptr(&self) -> Option<*mut u8> {
        match self {
            ZyntaxValue::Opaque { ptr, .. } => Some(*ptr),
            _ => None,
        }
    }

    /// Check if this opaque value owns its memory
    pub fn opaque_is_owned(&self) -> bool {
        match self {
            ZyntaxValue::Opaque { owned, .. } => *owned,
            _ => false,
        }
    }

    /// Reinterpret an opaque value as a typed reference
    ///
    /// # Safety
    /// - The opaque value must contain data of type `T`
    /// - The type `T` must have the same size and alignment as the stored data
    /// - The data must be valid for the lifetime of the returned reference
    pub unsafe fn opaque_as_ref<T>(&self) -> Option<&T> {
        match self {
            ZyntaxValue::Opaque { ptr, .. } if !ptr.is_null() => Some(&*(*ptr as *const T)),
            _ => None,
        }
    }

    /// Reinterpret an opaque value as a typed mutable reference
    ///
    /// # Safety
    /// - The opaque value must contain data of type `T`
    /// - The type `T` must have the same size and alignment as the stored data
    /// - The data must be valid for the lifetime of the returned reference
    /// - No other references to this data must exist
    pub unsafe fn opaque_as_mut<T>(&mut self) -> Option<&mut T> {
        match self {
            ZyntaxValue::Opaque { ptr, .. } if !ptr.is_null() => Some(&mut *(*ptr as *mut T)),
            _ => None,
        }
    }

    /// Convert from a DynamicValue
    ///
    /// # Safety
    /// The DynamicValue must have valid pointers
    pub unsafe fn from_dynamic(value: DynamicValue) -> ConversionResult<Self> {
        if value.is_null() {
            return Ok(ZyntaxValue::Null);
        }

        let type_id = value.type_id();
        let category = type_id.category();

        match category {
            TypeCategory::Void => Ok(ZyntaxValue::Void),

            TypeCategory::Bool => {
                if let Some(&v) = value.as_ref::<i32>() {
                    Ok(ZyntaxValue::Bool(v != 0))
                } else {
                    Ok(ZyntaxValue::Null)
                }
            }

            TypeCategory::Int => {
                // Try different int sizes based on type_id
                match type_id {
                    t if t == TypeId::I8 => {
                        value.as_ref::<i8>().map(|&v| ZyntaxValue::Int(v as i64))
                    }
                    t if t == TypeId::I16 => {
                        value.as_ref::<i16>().map(|&v| ZyntaxValue::Int(v as i64))
                    }
                    t if t == TypeId::I32 => {
                        value.as_ref::<i32>().map(|&v| ZyntaxValue::Int(v as i64))
                    }
                    _ => value.as_ref::<i64>().map(|&v| ZyntaxValue::Int(v)),
                }
                .ok_or(ConversionError::NullValue)
            }

            TypeCategory::UInt => match type_id {
                t if t == TypeId::U8 => value.as_ref::<u8>().map(|&v| ZyntaxValue::UInt(v as u64)),
                t if t == TypeId::U16 => {
                    value.as_ref::<u16>().map(|&v| ZyntaxValue::UInt(v as u64))
                }
                t if t == TypeId::U32 => {
                    value.as_ref::<u32>().map(|&v| ZyntaxValue::UInt(v as u64))
                }
                _ => value.as_ref::<u64>().map(|&v| ZyntaxValue::UInt(v)),
            }
            .ok_or(ConversionError::NullValue),

            TypeCategory::Float => {
                if type_id == TypeId::F32 {
                    value
                        .as_ref::<f32>()
                        .map(|&v| ZyntaxValue::Float(v as f64))
                        .ok_or(ConversionError::NullValue)
                } else {
                    value
                        .as_ref::<f64>()
                        .map(|&v| ZyntaxValue::Float(v))
                        .ok_or(ConversionError::NullValue)
                }
            }

            TypeCategory::String => {
                // Zyntax strings are length-prefixed: [i32 length][bytes...]
                let str_ptr = value.value_ptr as *const i32;
                if str_ptr.is_null() {
                    return Ok(ZyntaxValue::String(String::new()));
                }

                let length = *str_ptr;
                if length <= 0 {
                    return Ok(ZyntaxValue::String(String::new()));
                }

                let bytes_ptr = str_ptr.offset(1) as *const u8;
                let slice = std::slice::from_raw_parts(bytes_ptr, length as usize);
                let string = std::str::from_utf8(slice)?.to_string();
                Ok(ZyntaxValue::String(string))
            }

            TypeCategory::Array => {
                // Zyntax arrays: [i32 capacity][i32 length][elements...]
                let arr_ptr = value.value_ptr as *const i32;
                if arr_ptr.is_null() {
                    return Ok(ZyntaxValue::Array(Vec::new()));
                }

                let _capacity = *arr_ptr;
                let _length = *arr_ptr.offset(1);

                // For now, we'll return an opaque array since we don't know element types
                // A more complete implementation would use generic type args
                Ok(ZyntaxValue::Opaque {
                    type_meta: value.type_meta,
                    ptr: value.value_ptr,
                    owned: false, // Borrowed from DynamicValue, caller manages memory
                })
            }

            _ => {
                // For other types, return as opaque with full type metadata
                Ok(ZyntaxValue::Opaque {
                    type_meta: value.type_meta,
                    ptr: value.value_ptr,
                    owned: false, // Borrowed from DynamicValue, caller manages memory
                })
            }
        }
    }

    /// Convert to a DynamicValue (allocates on heap)
    pub fn into_dynamic(self) -> DynamicValue {
        match self {
            ZyntaxValue::Void | ZyntaxValue::Null => DynamicValue::null(),
            ZyntaxValue::Bool(v) => DynamicValue::from_bool(v),
            ZyntaxValue::Int(v) => {
                // Use i64 for large values, i32 for smaller ones
                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                    DynamicValue::from_i32(v as i32)
                } else {
                    DynamicValue::from_i64(v)
                }
            }
            ZyntaxValue::UInt(v) => {
                // Store as i64 for DynamicValue compatibility
                DynamicValue::from_i64(v as i64)
            }
            ZyntaxValue::Float(v) => DynamicValue::from_f64(v),
            ZyntaxValue::String(s) => {
                // Create length-prefixed string in Zyntax format
                let len = s.len() as i32;
                let total_size = std::mem::size_of::<i32>() + s.len();

                unsafe {
                    let ptr = std::alloc::alloc(
                        std::alloc::Layout::from_size_align(total_size, 4).unwrap(),
                    );

                    if ptr.is_null() {
                        return DynamicValue::null();
                    }

                    // Write length
                    *(ptr as *mut i32) = len;
                    // Write string bytes
                    std::ptr::copy_nonoverlapping(
                        s.as_ptr(),
                        ptr.offset(std::mem::size_of::<i32>() as isize),
                        s.len(),
                    );

                    DynamicValue::from_string(s)
                }
            }
            ZyntaxValue::Optional(inner) => match *inner {
                Some(v) => v.into_dynamic(),
                None => DynamicValue::null(),
            },
            // For complex types, we'd need more sophisticated handling
            _ => DynamicValue::null(),
        }
    }

    /// Create a struct value
    pub fn new_struct(type_name: impl Into<String>) -> StructBuilder {
        StructBuilder {
            type_name: type_name.into(),
            fields: HashMap::new(),
        }
    }

    /// Create an enum variant value
    pub fn new_enum(
        type_name: impl Into<String>,
        variant: impl Into<String>,
        data: Option<ZyntaxValue>,
    ) -> Self {
        ZyntaxValue::Enum {
            type_name: type_name.into(),
            variant: variant.into(),
            data: data.map(Box::new),
        }
    }
}

impl Default for ZyntaxValue {
    fn default() -> Self {
        ZyntaxValue::Null
    }
}

/// Builder for struct values
pub struct StructBuilder {
    type_name: String,
    fields: HashMap<String, ZyntaxValue>,
}

impl StructBuilder {
    /// Add a field to the struct
    pub fn field(mut self, name: impl Into<String>, value: impl Into<ZyntaxValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Build the struct value
    pub fn build(self) -> ZyntaxValue {
        ZyntaxValue::Struct {
            type_name: self.type_name,
            fields: self.fields,
        }
    }
}

// ============================================================================
// From implementations for convenient construction
// ============================================================================

impl From<bool> for ZyntaxValue {
    fn from(v: bool) -> Self {
        ZyntaxValue::Bool(v)
    }
}

impl From<i8> for ZyntaxValue {
    fn from(v: i8) -> Self {
        ZyntaxValue::Int(v as i64)
    }
}

impl From<i16> for ZyntaxValue {
    fn from(v: i16) -> Self {
        ZyntaxValue::Int(v as i64)
    }
}

impl From<i32> for ZyntaxValue {
    fn from(v: i32) -> Self {
        ZyntaxValue::Int(v as i64)
    }
}

impl From<i64> for ZyntaxValue {
    fn from(v: i64) -> Self {
        ZyntaxValue::Int(v)
    }
}

impl From<u8> for ZyntaxValue {
    fn from(v: u8) -> Self {
        ZyntaxValue::UInt(v as u64)
    }
}

impl From<u16> for ZyntaxValue {
    fn from(v: u16) -> Self {
        ZyntaxValue::UInt(v as u64)
    }
}

impl From<u32> for ZyntaxValue {
    fn from(v: u32) -> Self {
        ZyntaxValue::UInt(v as u64)
    }
}

impl From<u64> for ZyntaxValue {
    fn from(v: u64) -> Self {
        ZyntaxValue::UInt(v)
    }
}

impl From<f32> for ZyntaxValue {
    fn from(v: f32) -> Self {
        ZyntaxValue::Float(v as f64)
    }
}

impl From<f64> for ZyntaxValue {
    fn from(v: f64) -> Self {
        ZyntaxValue::Float(v)
    }
}

impl From<String> for ZyntaxValue {
    fn from(v: String) -> Self {
        ZyntaxValue::String(v)
    }
}

impl From<&str> for ZyntaxValue {
    fn from(v: &str) -> Self {
        ZyntaxValue::String(v.to_string())
    }
}

impl<T: Into<ZyntaxValue>> From<Vec<T>> for ZyntaxValue {
    fn from(v: Vec<T>) -> Self {
        ZyntaxValue::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<ZyntaxValue>> From<Option<T>> for ZyntaxValue {
    fn from(v: Option<T>) -> Self {
        ZyntaxValue::Optional(Box::new(v.map(Into::into)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_category() {
        assert_eq!(ZyntaxValue::Int(42).type_category(), TypeCategory::Int);
        assert_eq!(
            ZyntaxValue::String("hi".into()).type_category(),
            TypeCategory::String
        );
        assert_eq!(
            ZyntaxValue::Array(vec![]).type_category(),
            TypeCategory::Array
        );
    }

    #[test]
    fn test_accessors() {
        let int_val = ZyntaxValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(int_val.as_str(), None);

        let str_val = ZyntaxValue::String("hello".into());
        assert_eq!(str_val.as_str(), Some("hello"));
        assert_eq!(str_val.as_int(), None);
    }

    #[test]
    fn test_struct_builder() {
        let point = ZyntaxValue::new_struct("Point")
            .field("x", 10i32)
            .field("y", 20i32)
            .build();

        assert!(matches!(point, ZyntaxValue::Struct { .. }));
        assert_eq!(point.get_field("x"), Some(&ZyntaxValue::Int(10)));
        assert_eq!(point.get_field("y"), Some(&ZyntaxValue::Int(20)));
    }

    #[test]
    fn test_from_impls() {
        let v: ZyntaxValue = 42i32.into();
        assert!(matches!(v, ZyntaxValue::Int(42)));

        let v: ZyntaxValue = "hello".into();
        assert!(matches!(v, ZyntaxValue::String(s) if s == "hello"));

        let v: ZyntaxValue = vec![1i32, 2, 3].into();
        assert!(matches!(v, ZyntaxValue::Array(_)));
    }
}
