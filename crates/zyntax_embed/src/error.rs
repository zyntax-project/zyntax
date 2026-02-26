//! Error types for Zyntax value conversion

use thiserror::Error;
use zyntax_compiler::zrtl::{TypeCategory, TypeId};

/// Errors that can occur during value conversion
#[derive(Debug, Error)]
pub enum ConversionError {
    /// Type mismatch during conversion
    #[error("Type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch {
        expected: TypeCategory,
        found: TypeCategory,
    },

    /// Specific type ID mismatch
    #[error("Type ID mismatch: expected {expected:?}, found {found:?}")]
    TypeIdMismatch { expected: TypeId, found: TypeId },

    /// Null pointer where a value was expected
    #[error("Unexpected null value")]
    NullValue,

    /// Invalid UTF-8 in string conversion
    #[error("Invalid UTF-8 in string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    /// Invalid UTF-8 when converting from bytes
    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8String(#[from] std::string::FromUtf8Error),

    /// Array element conversion failed
    #[error("Array element conversion failed at index {index}: {source}")]
    ArrayElementError {
        index: usize,
        #[source]
        source: Box<ConversionError>,
    },

    /// Struct field conversion failed
    #[error("Struct field '{field}' conversion failed: {source}")]
    StructFieldError {
        field: String,
        #[source]
        source: Box<ConversionError>,
    },

    /// Integer overflow during conversion
    #[error("Integer overflow: value {value} doesn't fit in target type")]
    IntegerOverflow { value: i128 },

    /// Float precision loss warning (not fatal)
    #[error("Float precision loss converting {from} to {to}")]
    FloatPrecisionLoss {
        from: &'static str,
        to: &'static str,
    },

    /// Missing generic type arguments
    #[error("Missing generic type arguments for {type_name}")]
    MissingGenericArgs { type_name: String },

    /// Unsupported type conversion
    #[error("Unsupported conversion from {from:?} to {to}")]
    UnsupportedConversion { from: TypeCategory, to: String },

    /// Memory allocation failed
    #[error("Memory allocation failed")]
    AllocationFailed,
}

/// General errors for Zyntax embedding operations
#[derive(Debug, Error)]
pub enum ZyntaxError {
    /// Conversion error
    #[error("Conversion error: {0}")]
    Conversion(#[from] ConversionError),

    /// Runtime error from Zyntax execution
    #[error("Runtime error: {message}")]
    Runtime { message: String },

    /// Plugin loading error
    #[error("Plugin error: {message}")]
    Plugin { message: String },

    /// Compilation error
    #[error("Compilation error: {message}")]
    Compilation { message: String },
}

impl ConversionError {
    /// Create a type mismatch error
    pub fn type_mismatch(expected: TypeCategory, found: TypeCategory) -> Self {
        Self::TypeMismatch { expected, found }
    }

    /// Create a type ID mismatch error
    pub fn type_id_mismatch(expected: TypeId, found: TypeId) -> Self {
        Self::TypeIdMismatch { expected, found }
    }

    /// Create an array element error
    pub fn array_element(index: usize, source: ConversionError) -> Self {
        Self::ArrayElementError {
            index,
            source: Box::new(source),
        }
    }

    /// Create a struct field error
    pub fn struct_field(field: impl Into<String>, source: ConversionError) -> Self {
        Self::StructFieldError {
            field: field.into(),
            source: Box::new(source),
        }
    }
}

/// Result type for conversion operations
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Result type for general Zyntax operations
pub type ZyntaxResult<T> = Result<T, ZyntaxError>;
