//! # AST Error Types
//!
//! Error handling for AST construction and validation.

use crate::source::Span;
use thiserror::Error;

/// Main error type for AST building operations
#[derive(Debug, Error)]
pub enum AstError {
    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Invalid node type: expected {expected}, found {found}")]
    InvalidNodeType { expected: String, found: String },

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Type error: {message}")]
    TypeError { message: String, span: Option<Span> },

    #[error("Semantic error: {message}")]
    SemanticError { message: String, span: Option<Span> },

    #[error("Internal error: {0}")]
    Internal(String),
}

impl AstError {
    pub fn type_error(message: impl Into<String>, span: Option<Span>) -> Self {
        Self::TypeError {
            message: message.into(),
            span,
        }
    }

    pub fn semantic_error(message: impl Into<String>, span: Option<Span>) -> Self {
        Self::SemanticError {
            message: message.into(),
            span,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn span(&self) -> Option<Span> {
        match self {
            AstError::TypeError { span, .. } | AstError::SemanticError { span, .. } => *span,
            AstError::Validation(validation_error) => validation_error.span(),
            _ => None,
        }
    }
}

/// Validation errors for AST nodes
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Invalid identifier: {name}")]
    InvalidIdentifier { name: String, span: Span },

    #[error("Invalid literal: {message}")]
    InvalidLiteral { message: String, span: Span },

    #[error("Invalid function signature: {message}")]
    InvalidFunctionSignature { message: String, span: Span },

    #[error("Invalid type: {message}")]
    InvalidType { message: String, span: Span },

    #[error("Invalid pattern: {message}")]
    InvalidPattern { message: String, span: Span },

    #[error("Missing return type for function")]
    MissingReturnType { span: Span },

    #[error("Duplicate parameter name: {name}")]
    DuplicateParameter { name: String, span: Span },

    #[error("Invalid visibility modifier")]
    InvalidVisibility { span: Span },

    #[error("Inconsistent node structure: {message}")]
    InconsistentStructure { message: String, span: Span },

    #[error("Unsupported feature: {feature}")]
    UnsupportedFeature { feature: String, span: Span },
}

impl ValidationError {
    pub fn invalid_identifier(name: impl Into<String>, span: Span) -> Self {
        Self::InvalidIdentifier {
            name: name.into(),
            span,
        }
    }

    pub fn invalid_literal(message: impl Into<String>, span: Span) -> Self {
        Self::InvalidLiteral {
            message: message.into(),
            span,
        }
    }

    pub fn invalid_function_signature(message: impl Into<String>, span: Span) -> Self {
        Self::InvalidFunctionSignature {
            message: message.into(),
            span,
        }
    }

    pub fn invalid_type(message: impl Into<String>, span: Span) -> Self {
        Self::InvalidType {
            message: message.into(),
            span,
        }
    }

    pub fn invalid_pattern(message: impl Into<String>, span: Span) -> Self {
        Self::InvalidPattern {
            message: message.into(),
            span,
        }
    }

    pub fn missing_return_type(span: Span) -> Self {
        Self::MissingReturnType { span }
    }

    pub fn duplicate_parameter(name: impl Into<String>, span: Span) -> Self {
        Self::DuplicateParameter {
            name: name.into(),
            span,
        }
    }

    pub fn invalid_visibility(span: Span) -> Self {
        Self::InvalidVisibility { span }
    }

    pub fn inconsistent_structure(message: impl Into<String>, span: Span) -> Self {
        Self::InconsistentStructure {
            message: message.into(),
            span,
        }
    }

    pub fn unsupported_feature(feature: impl Into<String>, span: Span) -> Self {
        Self::UnsupportedFeature {
            feature: feature.into(),
            span,
        }
    }

    pub fn span(&self) -> Option<Span> {
        match self {
            ValidationError::InvalidIdentifier { span, .. }
            | ValidationError::InvalidLiteral { span, .. }
            | ValidationError::InvalidFunctionSignature { span, .. }
            | ValidationError::InvalidType { span, .. }
            | ValidationError::InvalidPattern { span, .. }
            | ValidationError::MissingReturnType { span }
            | ValidationError::DuplicateParameter { span, .. }
            | ValidationError::InvalidVisibility { span }
            | ValidationError::InconsistentStructure { span, .. }
            | ValidationError::UnsupportedFeature { span, .. } => Some(*span),
        }
    }
}

/// Error reporter for collecting and displaying multiple errors
#[derive(Debug, Default)]
pub struct ErrorReporter {
    errors: Vec<AstError>,
    warnings: Vec<Warning>,
    max_errors: usize,
}

impl ErrorReporter {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            max_errors: 100,
        }
    }

    pub fn with_max_errors(max_errors: usize) -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            max_errors,
        }
    }

    /// Report an error
    pub fn error(&mut self, error: AstError) -> Result<(), AstError> {
        self.errors.push(error);

        if self.errors.len() >= self.max_errors {
            Err(AstError::internal("Too many errors"))
        } else {
            Ok(())
        }
    }

    /// Report a warning
    pub fn warning(&mut self, warning: Warning) {
        self.warnings.push(warning);
    }

    /// Get all errors
    pub fn errors(&self) -> &[AstError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get total error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get total warning count
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// Clear all errors and warnings
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }

    /// Clear only errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Clear only warnings
    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }

    /// Generate a comprehensive error report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        if !self.errors.is_empty() {
            report.push_str(&format!("Found {} error(s):\n", self.errors.len()));
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("  {}: {}\n", i + 1, error));
            }
            report.push('\n');
        }

        if !self.warnings.is_empty() {
            report.push_str(&format!("Found {} warning(s):\n", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("  {}: {}\n", i + 1, warning));
            }
            report.push('\n');
        }

        if self.errors.is_empty() && self.warnings.is_empty() {
            report.push_str("No errors or warnings to report.\n");
        }

        report
    }
}

/// Warning types for non-fatal issues
#[derive(Debug, Clone)]
pub struct Warning {
    pub kind: WarningKind,
    pub message: String,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningKind {
    UnusedVariable,
    UnusedFunction,
    UnusedImport,
    DeadCode,
    MissingDocumentation,
    DeprecatedFeature,
    StyleIssue,
    Performance,
    Other,
}

impl Warning {
    pub fn new(kind: WarningKind, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            kind,
            message: message.into(),
            span,
        }
    }

    pub fn unused_variable(name: impl Into<String>, span: Span) -> Self {
        Self::new(
            WarningKind::UnusedVariable,
            format!("unused variable: {}", name.into()),
            Some(span),
        )
    }

    pub fn unused_function(name: impl Into<String>, span: Span) -> Self {
        Self::new(
            WarningKind::UnusedFunction,
            format!("unused function: {}", name.into()),
            Some(span),
        )
    }

    pub fn dead_code(span: Span) -> Self {
        Self::new(
            WarningKind::DeadCode,
            "unreachable code".to_string(),
            Some(span),
        )
    }

    pub fn missing_documentation(item: impl Into<String>, span: Span) -> Self {
        Self::new(
            WarningKind::MissingDocumentation,
            format!("missing documentation for {}", item.into()),
            Some(span),
        )
    }

    pub fn deprecated_feature(feature: impl Into<String>, span: Span) -> Self {
        Self::new(
            WarningKind::DeprecatedFeature,
            format!("use of deprecated feature: {}", feature.into()),
            Some(span),
        )
    }
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "warning: {}", self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Span;

    #[test]
    fn test_ast_error_creation() {
        let error = AstError::MissingRequiredField("test".to_string());
        assert!(error.to_string().contains("test"));
    }

    #[test]
    fn test_validation_error_creation() {
        let span = Span::new(0, 5);
        let error = ValidationError::invalid_identifier("test", span);

        assert_eq!(error.span(), Some(span));
        assert!(error.to_string().contains("test"));
    }

    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new();

        assert!(!reporter.has_errors());
        assert_eq!(reporter.error_count(), 0);

        let error = AstError::MissingRequiredField("test".to_string());
        reporter.error(error).unwrap();

        assert!(reporter.has_errors());
        assert_eq!(reporter.error_count(), 1);

        let report = reporter.generate_report();
        assert!(report.contains("Found 1 error"));
    }

    #[test]
    fn test_warning_creation() {
        let span = Span::new(0, 5);
        let warning = Warning::unused_variable("x", span);

        assert_eq!(warning.kind, WarningKind::UnusedVariable);
        assert!(warning.message.contains("unused variable: x"));
        assert_eq!(warning.span, Some(span));
    }
}
