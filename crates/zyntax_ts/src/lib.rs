//! TypeScript source to Zyntax's typed AST, with compile-time release analysis.

mod ambient;
mod lower;
mod modules;
mod project;
mod runtime;

pub use lower::{parse_module, parse_module_with_dependencies, parse_program};
pub use modules::{ModuleDependency, ModuleId};
pub use project::{
    CachePreparation, CacheStatus, ProjectError, SourceProject, load_project, load_project_cached,
    load_project_for_runtime, load_project_for_runtime_background,
};
pub use runtime::register_runtime;

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
    diagnostic: Option<Box<zyntax_typed_ast::diagnostics::Diagnostic>>,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>, span: oxc_span::Span) -> Self {
        Self {
            kind,
            message: message.into(),
            start: span.start as usize,
            end: span.end as usize,
            diagnostic: None,
        }
    }

    pub(crate) fn from_oxc(kind: ErrorKind, error: &oxc_diagnostics::OxcDiagnostic) -> Self {
        use zyntax_typed_ast::Span;
        use zyntax_typed_ast::diagnostics::Diagnostic;
        let primary = error
            .labels
            .iter()
            .position(|label| label.primary())
            .unwrap_or(0);
        let at = error
            .labels
            .get(primary)
            .map_or(oxc_span::Span::default(), |label| label.span());
        let mut diagnostic = Diagnostic::error(error.to_string());
        for (index, label) in error.labels.iter().enumerate() {
            let span = Span::new(
                label.offset() as usize,
                (label.offset() + label.len()) as usize,
            );
            let message = label.label().unwrap_or("here");
            diagnostic = if index == primary {
                diagnostic.with_primary(span, message)
            } else {
                diagnostic.with_secondary(span, message)
            };
        }
        if let Some(help) = &error.help {
            diagnostic = diagnostic.with_help(help.as_ref());
        }
        if let Some(note) = &error.note {
            diagnostic = diagnostic.with_note(note.as_ref());
        }
        let mut result = Self::new(kind, diagnostic.message.clone(), at);
        result.diagnostic = Some(Box::new(diagnostic));
        result
    }

    /// Render with the compiler's shared Ariadne source formatter.
    pub fn render(&self, file: &str, source: &str, use_colors: bool) -> String {
        use zyntax_typed_ast::Span;
        use zyntax_typed_ast::diagnostics::{Diagnostic, render_diagnostic};
        let diagnostic;
        let diagnostic = if let Some(details) = &self.diagnostic {
            details.as_ref()
        } else {
            let label = match self.kind {
                ErrorKind::Syntax => "syntax error",
                ErrorKind::Type => "type error",
                ErrorKind::Soundness => "additional soundness check",
                ErrorKind::Unsupported => "native support is not implemented yet",
            };
            diagnostic = Diagnostic::error(&self.message)
                .with_primary(Span::new(self.start, self.end), label);
            &diagnostic
        };
        render_diagnostic(diagnostic, file, source, use_colors)
    }
}
