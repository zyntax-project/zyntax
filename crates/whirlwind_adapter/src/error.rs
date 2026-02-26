//! Error types for the Whirlwind adapter

use thiserror::Error;

/// Result type for adapter operations
pub type AdapterResult<T> = Result<T, AdapterError>;

/// Errors that can occur during Whirlwind → TypedAST conversion
#[derive(Debug, Error)]
pub enum AdapterError {
    /// Type conversion failed
    #[error("Type conversion error: {0}")]
    TypeConversion(String),

    /// Expression conversion failed
    #[error("Expression conversion error: {0}")]
    ExpressionConversion(String),

    /// Statement conversion failed
    #[error("Statement conversion error: {0}")]
    StatementConversion(String),

    /// Unsupported feature in Whirlwind source
    #[error("Unsupported Whirlwind feature: {0}")]
    UnsupportedFeature(String),

    /// Symbol not found in symbol table
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),

    /// Type mismatch during conversion
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    /// Invalid Whirlwind IR structure
    #[error("Invalid Whirlwind IR: {0}")]
    InvalidIR(String),

    /// Module not found
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    /// Generic error with context
    #[error("Adapter error: {0}")]
    Generic(String),
}

impl AdapterError {
    /// Create a type conversion error
    pub fn type_conversion(msg: impl Into<String>) -> Self {
        Self::TypeConversion(msg.into())
    }

    /// Create an expression conversion error
    pub fn expression_conversion(msg: impl Into<String>) -> Self {
        Self::ExpressionConversion(msg.into())
    }

    /// Create a statement conversion error
    pub fn statement_conversion(msg: impl Into<String>) -> Self {
        Self::StatementConversion(msg.into())
    }

    /// Create an unsupported feature error
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::UnsupportedFeature(msg.into())
    }

    /// Create a symbol not found error
    pub fn symbol_not_found(symbol: impl Into<String>) -> Self {
        Self::SymbolNotFound(symbol.into())
    }
}
