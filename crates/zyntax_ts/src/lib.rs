//! TypeScript source to Zyntax's typed AST, with compile-time release analysis.

mod lower;
mod runtime;

pub use lower::parse_program;
pub use runtime::register_runtime;

/// The module body the host enters.
pub const ENTRY: &str = "ts$main";

/// Additional checks beyond TypeScript's strict static checking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Soundness {
    /// Match TypeScript's assignment rules for supported constructs.
    #[default]
    TypeScript,
    /// Reject weakening readonly views into writable aliases.
    Checked,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub soundness: Soundness,
}

/// Diagnostic category, kept separate from an unimplemented language feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
    Type,
    Soundness,
    Unsupported,
}

#[derive(Debug, thiserror::Error)]
#[error("{kind:?}: {message} at {start}..{end}")]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub start: usize,
    pub end: usize,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>, span: oxc_span::Span) -> Self {
        Self {
            kind,
            message: message.into(),
            start: span.start as usize,
            end: span.end as usize,
        }
    }

    pub fn render(&self, file: &str, source: &str) -> String {
        let at = self.start.min(source.len());
        let line = source[..at].bytes().filter(|&b| b == b'\n').count() + 1;
        let column = source[..at]
            .rsplit('\n')
            .next()
            .unwrap_or("")
            .chars()
            .count()
            + 1;
        format!("{file}:{line}:{column}: {:?}: {}", self.kind, self.message)
    }
}
