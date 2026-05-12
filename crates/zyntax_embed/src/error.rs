//! Error types for Zyntax embedding.
//!
//! `ConversionError` and `ConversionResult` were moved to
//! [`zyntax_compiler::value`] alongside `ZyntaxValue` itself so both
//! crates can use them. They're re-exported here for backwards-compat.
//! `ZyntaxError` (the higher-level embed error) remains local.

use thiserror::Error;

pub use zyntax_compiler::value::{ConversionError, ConversionResult};

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

/// Result type for general Zyntax operations
pub type ZyntaxResult<T> = Result<T, ZyntaxError>;
