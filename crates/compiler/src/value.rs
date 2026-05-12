//! `ZyntaxValue` — the single value type used by every layer of the
//! runtime.
//!
//! Lives in `zyntax_compiler` (rather than `zyntax_embed`) so the BC
//! interpreter can use it directly without a separate `InterpValue`
//! type and a lossy conversion at the API boundary. The `Conversion
//! Error` companion type lives here too so the `from_dynamic` /
//! `into_dynamic` marshalling methods can live alongside `ZyntaxValue`
//! itself.
//!
//! `zyntax_embed` re-exports both for source-compat with embedders
//! that depend on `zyntax_embed::{ZyntaxValue, ConversionError}`.

use std::collections::HashMap;
use thiserror::Error;

use crate::zrtl::{DynamicValue, TypeCategory, TypeId, TypeMeta};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during value conversion (DynamicValue ↔
/// `ZyntaxValue`, Rust types ↔ `ZyntaxValue`).
#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("Type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch {
        expected: TypeCategory,
        found: TypeCategory,
    },

    #[error("Type ID mismatch: expected {expected:?}, found {found:?}")]
    TypeIdMismatch { expected: TypeId, found: TypeId },

    #[error("Unexpected null value")]
    NullValue,

    #[error("Invalid UTF-8 in string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8String(#[from] std::string::FromUtf8Error),

    #[error("Array element conversion failed at index {index}: {source}")]
    ArrayElementError {
        index: usize,
        #[source]
        source: Box<ConversionError>,
    },

    #[error("Struct field '{field}' conversion failed: {source}")]
    StructFieldError {
        field: String,
        #[source]
        source: Box<ConversionError>,
    },

    #[error("Integer overflow: value {value} doesn't fit in target type")]
    IntegerOverflow { value: i128 },

    #[error("Float precision loss converting {from} to {to}")]
    FloatPrecisionLoss {
        from: &'static str,
        to: &'static str,
    },

    #[error("Missing generic type arguments for {type_name}")]
    MissingGenericArgs { type_name: String },

    #[error("Unsupported conversion from {from:?} to {to}")]
    UnsupportedConversion { from: TypeCategory, to: String },

    #[error("Memory allocation failed")]
    AllocationFailed,
}

impl ConversionError {
    pub fn type_mismatch(expected: TypeCategory, found: TypeCategory) -> Self {
        Self::TypeMismatch { expected, found }
    }

    pub fn type_id_mismatch(expected: TypeId, found: TypeId) -> Self {
        Self::TypeIdMismatch { expected, found }
    }

    pub fn array_element(index: usize, source: ConversionError) -> Self {
        ConversionError::ArrayElementError {
            index,
            source: Box::new(source),
        }
    }

    pub fn struct_field(field: impl Into<String>, source: ConversionError) -> Self {
        ConversionError::StructFieldError {
            field: field.into(),
            source: Box::new(source),
        }
    }
}

pub type ConversionResult<T> = Result<T, ConversionError>;

// ─────────────────────────────────────────────────────────────────────────────
// ZyntaxValue
// ─────────────────────────────────────────────────────────────────────────────

/// The single runtime value type. Used by the BC interpreter, by the
/// host embedding API, and by FFI marshalling.
///
/// Comes in two flavours of numeric variants:
/// - **Generic** (`Int(i64)`, `UInt(u64)`, `Float(f64)`) — host-friendly,
///   width-agnostic. Used by `From<T>` impls and most embedding code
///   that doesn't care about the exact integer width.
/// - **Width-precise siblings** (`I8`, `I16`, `I32`, `U8`, `U16`, `U32`,
///   `F32`) — the BC interpreter emits these for functions whose HIR
///   signature has a narrow width, so the host gets back the exact
///   width info instead of a widened `Int(i64)`. Width-agnostic
///   accessors (`as_int` / `as_uint` / `as_float`) accept both.
///
/// `i64` / `u64` / `f64` results reuse the generic variants (`Int` /
/// `UInt` / `Float`) — there's no separate `I64` / `U64` / `F64`
/// since they'd hold the same payload as the generic ones.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ZyntaxValue {
    /// Void/unit type (no value)
    Void,

    /// Null value
    Null,

    /// Boolean value
    Bool(bool),

    /// Signed integer (stored as i64 for flexibility). Generic
    /// variant — `From<i32>`, `From<i64>` etc. all produce this.
    /// The BC interpreter emits this for functions returning `i64`.
    Int(i64),

    /// Unsigned integer (stored as u64 for flexibility). Generic
    /// variant. Interpreter emits this for `u64`.
    UInt(u64),

    /// Floating point (stored as f64 for precision). Generic variant.
    /// Interpreter emits this for `f64`.
    Float(f64),

    // ── Width-precise siblings ──
    /// Signed 8-bit integer (precise width).
    I8(i8),
    /// Signed 16-bit integer (precise width).
    I16(i16),
    /// Signed 32-bit integer (precise width).
    I32(i32),
    /// Unsigned 8-bit integer (precise width).
    U8(u8),
    /// Unsigned 16-bit integer (precise width).
    U16(u16),
    /// Unsigned 32-bit integer (precise width).
    U32(u32),
    /// 32-bit float (precise width).
    F32(f32),

    /// String value (owned, UTF-8)
    String(String),

    /// Array of values
    Array(Vec<ZyntaxValue>),

    /// Map/Dictionary (String keys for simplicity)
    Map(HashMap<String, ZyntaxValue>),

    /// Struct with named fields
    Struct {
        type_name: String,
        fields: HashMap<String, ZyntaxValue>,
    },

    /// Enum variant
    Enum {
        type_name: String,
        variant: String,
        data: Option<Box<ZyntaxValue>>,
    },

    /// Optional value (Some or None)
    Optional(Box<Option<ZyntaxValue>>),

    /// Result value (Ok or Err)
    Result(Box<Result<ZyntaxValue, ZyntaxValue>>),

    /// Tuple / positional aggregate. Also serves as the interpreter's
    /// flat struct representation — the BC interp uses positional
    /// (HIR-index) access into the inner `Vec`, matching how HIR's
    /// `InsertValue` / `ExtractValue` ops work.
    Tuple(Vec<ZyntaxValue>),

    /// Function reference (stored as opaque pointer)
    Function { ptr: usize, name: Option<String> },

    /// Raw pointer (for FFI / interp memory).
    Pointer(*mut u8),

    /// Opaque/Dynamic value (for types we can't introspect)
    Opaque {
        type_meta: *const TypeMeta,
        ptr: *mut u8,
        owned: bool,
    },

    /// SSA-defined-but-not-yet-set sentinel. The BC interpreter uses
    /// this for unbound register slots; should never escape to host
    /// code. Distinct from `Null` so the interpreter can tell "no
    /// value yet" from "explicit null".
    Undef,
}

// PartialEq with raw-pointer handling — pointers compare by address,
// Undef equals only Undef.
impl PartialEq for ZyntaxValue {
    fn eq(&self, other: &Self) -> bool {
        use ZyntaxValue::*;
        match (self, other) {
            (Void, Void) | (Null, Null) | (Undef, Undef) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (UInt(a), UInt(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (I8(a), I8(b)) => a == b,
            (I16(a), I16(b)) => a == b,
            (I32(a), I32(b)) => a == b,
            (U8(a), U8(b)) => a == b,
            (U16(a), U16(b)) => a == b,
            (U32(a), U32(b)) => a == b,
            (F32(a), F32(b)) => a == b,
            (String(a), String(b)) => a == b,
            (Array(a), Array(b)) => a == b,
            (Map(a), Map(b)) => a == b,
            (
                Struct {
                    type_name: t1,
                    fields: f1,
                },
                Struct {
                    type_name: t2,
                    fields: f2,
                },
            ) => t1 == t2 && f1 == f2,
            (
                Enum {
                    type_name: t1,
                    variant: v1,
                    data: d1,
                },
                Enum {
                    type_name: t2,
                    variant: v2,
                    data: d2,
                },
            ) => t1 == t2 && v1 == v2 && d1 == d2,
            (Optional(a), Optional(b)) => a == b,
            (Result(a), Result(b)) => a == b,
            (Tuple(a), Tuple(b)) => a == b,
            (Function { ptr: p1, name: n1 }, Function { ptr: p2, name: n2 }) => {
                p1 == p2 && n1 == n2
            }
            (Pointer(a), Pointer(b)) => std::ptr::eq(*a, *b),
            (
                Opaque {
                    type_meta: t1,
                    ptr: p1,
                    owned: o1,
                },
                Opaque {
                    type_meta: t2,
                    ptr: p2,
                    owned: o2,
                },
            ) => std::ptr::eq(*t1, *t2) && std::ptr::eq(*p1, *p2) && o1 == o2,
            _ => false,
        }
    }
}

// SAFETY: `ZyntaxValue` contains raw pointers in some variants; we
// manage them carefully. For `Pointer` and `Opaque` variants, the
// caller is responsible for memory safety.
unsafe impl Send for ZyntaxValue {}
unsafe impl Sync for ZyntaxValue {}

impl ZyntaxValue {
    pub fn type_category(&self) -> TypeCategory {
        match self {
            ZyntaxValue::Void | ZyntaxValue::Null | ZyntaxValue::Undef => TypeCategory::Void,
            ZyntaxValue::Bool(_) => TypeCategory::Bool,
            ZyntaxValue::Int(_)
            | ZyntaxValue::I8(_)
            | ZyntaxValue::I16(_)
            | ZyntaxValue::I32(_) => TypeCategory::Int,
            ZyntaxValue::UInt(_)
            | ZyntaxValue::U8(_)
            | ZyntaxValue::U16(_)
            | ZyntaxValue::U32(_) => TypeCategory::UInt,
            ZyntaxValue::Float(_) | ZyntaxValue::F32(_) => TypeCategory::Float,
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

    pub fn is_null_or_void(&self) -> bool {
        matches!(
            self,
            ZyntaxValue::Void | ZyntaxValue::Null | ZyntaxValue::Undef
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            ZyntaxValue::Int(_)
                | ZyntaxValue::UInt(_)
                | ZyntaxValue::I8(_)
                | ZyntaxValue::I16(_)
                | ZyntaxValue::I32(_)
                | ZyntaxValue::U8(_)
                | ZyntaxValue::U16(_)
                | ZyntaxValue::U32(_)
        )
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || matches!(self, ZyntaxValue::Float(_) | ZyntaxValue::F32(_))
    }

    /// Width-agnostic signed-integer accessor. Returns `Some(i64)`
    /// for the generic `Int` variant AND for any of the width-precise
    /// signed siblings (I8/I16/I32). Unsigned variants are accepted
    /// only when they fit in `i64::MAX`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ZyntaxValue::Int(v) => Some(*v),
            ZyntaxValue::I8(v) => Some(*v as i64),
            ZyntaxValue::I16(v) => Some(*v as i64),
            ZyntaxValue::I32(v) => Some(*v as i64),
            ZyntaxValue::UInt(v) if *v <= i64::MAX as u64 => Some(*v as i64),
            ZyntaxValue::U8(v) => Some(*v as i64),
            ZyntaxValue::U16(v) => Some(*v as i64),
            ZyntaxValue::U32(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        self.as_int().and_then(|v| {
            if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                Some(v as i32)
            } else {
                None
            }
        })
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_int()
    }

    /// Width-agnostic unsigned-integer accessor. Accepts UInt, U8,
    /// U16, U32, plus non-negative signed variants.
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            ZyntaxValue::UInt(v) => Some(*v),
            ZyntaxValue::U8(v) => Some(*v as u64),
            ZyntaxValue::U16(v) => Some(*v as u64),
            ZyntaxValue::U32(v) => Some(*v as u64),
            ZyntaxValue::Int(v) if *v >= 0 => Some(*v as u64),
            ZyntaxValue::I8(v) if *v >= 0 => Some(*v as u64),
            ZyntaxValue::I16(v) if *v >= 0 => Some(*v as u64),
            ZyntaxValue::I32(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Width-agnostic float accessor. Accepts Float, F32, and any
    /// integer variant (widening to f64).
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ZyntaxValue::Float(v) => Some(*v),
            ZyntaxValue::F32(v) => Some(*v as f64),
            _ => self.as_int().map(|i| i as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ZyntaxValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ZyntaxValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[ZyntaxValue]> {
        match self {
            ZyntaxValue::Array(arr) | ZyntaxValue::Tuple(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }

    pub fn get_field(&self, name: &str) -> Option<&ZyntaxValue> {
        match self {
            ZyntaxValue::Struct { fields, .. } => fields.get(name),
            ZyntaxValue::Map(map) => map.get(name),
            _ => None,
        }
    }

    pub fn is_opaque(&self) -> bool {
        matches!(self, ZyntaxValue::Opaque { .. })
    }

    pub fn opaque_type_meta(&self) -> Option<&TypeMeta> {
        match self {
            ZyntaxValue::Opaque { type_meta, .. } if !type_meta.is_null() => unsafe {
                Some(&**type_meta)
            },
            _ => None,
        }
    }

    pub fn opaque_type_id(&self) -> Option<TypeId> {
        self.opaque_type_meta().map(|meta| meta.type_id)
    }

    pub fn opaque_ptr(&self) -> Option<*mut u8> {
        match self {
            ZyntaxValue::Opaque { ptr, .. } => Some(*ptr),
            _ => None,
        }
    }

    pub fn opaque_is_owned(&self) -> bool {
        match self {
            ZyntaxValue::Opaque { owned, .. } => *owned,
            _ => false,
        }
    }

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

    /// # Safety
    /// The DynamicValue must have valid pointers.
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

            TypeCategory::Int => match type_id {
                t if t == TypeId::I8 => value.as_ref::<i8>().map(|&v| ZyntaxValue::Int(v as i64)),
                t if t == TypeId::I16 => value.as_ref::<i16>().map(|&v| ZyntaxValue::Int(v as i64)),
                t if t == TypeId::I32 => value.as_ref::<i32>().map(|&v| ZyntaxValue::Int(v as i64)),
                _ => value.as_ref::<i64>().map(|&v| ZyntaxValue::Int(v)),
            }
            .ok_or(ConversionError::NullValue),

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
                let arr_ptr = value.value_ptr as *const i32;
                if arr_ptr.is_null() {
                    return Ok(ZyntaxValue::Array(Vec::new()));
                }
                let _capacity = *arr_ptr;
                let _length = *arr_ptr.offset(1);

                Ok(ZyntaxValue::Opaque {
                    type_meta: value.type_meta,
                    ptr: value.value_ptr,
                    owned: false,
                })
            }

            _ => Ok(ZyntaxValue::Opaque {
                type_meta: value.type_meta,
                ptr: value.value_ptr,
                owned: false,
            }),
        }
    }

    pub fn into_dynamic(self) -> DynamicValue {
        match self {
            ZyntaxValue::Void | ZyntaxValue::Null | ZyntaxValue::Undef => DynamicValue::null(),
            ZyntaxValue::Bool(v) => DynamicValue::from_bool(v),
            ZyntaxValue::Int(v) => {
                if v >= i32::MIN as i64 && v <= i32::MAX as i64 {
                    DynamicValue::from_i32(v as i32)
                } else {
                    DynamicValue::from_i64(v)
                }
            }
            ZyntaxValue::UInt(v) => DynamicValue::from_i64(v as i64),
            ZyntaxValue::Float(v) => DynamicValue::from_f64(v),
            ZyntaxValue::String(s) => DynamicValue::from_string(s),
            ZyntaxValue::Optional(inner) => match *inner {
                Some(v) => v.into_dynamic(),
                None => DynamicValue::null(),
            },
            _ => DynamicValue::null(),
        }
    }

    pub fn new_struct(type_name: impl Into<String>) -> StructBuilder {
        StructBuilder {
            type_name: type_name.into(),
            fields: HashMap::new(),
        }
    }

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
    pub fn field(mut self, name: impl Into<String>, value: impl Into<ZyntaxValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    pub fn build(self) -> ZyntaxValue {
        ZyntaxValue::Struct {
            type_name: self.type_name,
            fields: self.fields,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// From impls for ergonomic construction
// ─────────────────────────────────────────────────────────────────────────────

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
