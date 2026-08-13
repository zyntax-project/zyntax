//! # Diagnostic Reporting System
//!
//! Rust-like compiler diagnostic reporting system with:
//! - Source location tracking and highlighting
//! - Error codes and structured messages
//! - Help text, hints, and suggestions
//! - Multi-span annotations
//! - Display formatting traits
//! - Integration with all compiler phases

use crate::arena::InternedString;
use crate::source::{SourceFile, SourceMap, Span};
use std::collections::HashMap;
use std::fmt;

/// Diagnostic severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    /// Internal compiler error
    Ice,
    /// Fatal error that stops compilation
    Error,
    /// Warning that doesn't stop compilation
    Warning,
    /// Note/informational message
    Note,
    /// Help/suggestion message
    Help,
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticLevel::Ice => write!(f, "internal compiler error"),
            DiagnosticLevel::Error => write!(f, "error"),
            DiagnosticLevel::Warning => write!(f, "warning"),
            DiagnosticLevel::Note => write!(f, "note"),
            DiagnosticLevel::Help => write!(f, "help"),
        }
    }
}

/// Diagnostic error codes (similar to rustc error codes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiagnosticCode(pub &'static str);

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Common diagnostic codes
pub mod codes {
    use super::DiagnosticCode;

    // Type checking errors
    pub const E0001: DiagnosticCode = DiagnosticCode("E0001"); // Type mismatch
    pub const E0002: DiagnosticCode = DiagnosticCode("E0002"); // Undefined variable
    pub const E0003: DiagnosticCode = DiagnosticCode("E0003"); // Too many arguments
    pub const E0004: DiagnosticCode = DiagnosticCode("E0004"); // Too few arguments
    pub const E0005: DiagnosticCode = DiagnosticCode("E0005"); // Not assignable
    pub const E0006: DiagnosticCode = DiagnosticCode("E0006"); // Unknown parameter
    pub const E0007: DiagnosticCode = DiagnosticCode("E0007"); // Named args not allowed
    pub const E0008: DiagnosticCode = DiagnosticCode("E0008"); // Field not found

    // Trait and generic errors
    pub const E0277: DiagnosticCode = DiagnosticCode("E0277"); // Trait not implemented
    pub const E0405: DiagnosticCode = DiagnosticCode("E0405"); // Cannot find trait

    // Ownership and borrowing errors
    pub const E0100: DiagnosticCode = DiagnosticCode("E0100"); // Use after move
    pub const E0101: DiagnosticCode = DiagnosticCode("E0101"); // Conflicting borrow
    pub const E0102: DiagnosticCode = DiagnosticCode("E0102"); // Double free

    // Lifetime errors
    pub const E0200: DiagnosticCode = DiagnosticCode("E0200"); // Lifetime constraint violation
    pub const E0201: DiagnosticCode = DiagnosticCode("E0201"); // Dangling reference

    // Control flow errors
    pub const E0300: DiagnosticCode = DiagnosticCode("E0300"); // Unreachable code
    pub const E0301: DiagnosticCode = DiagnosticCode("E0301"); // Missing return
    pub const E0302: DiagnosticCode = DiagnosticCode("E0302"); // Invalid break/continue

    // Lifetime errors
    pub const E0309: DiagnosticCode = DiagnosticCode("E0309"); // Lifetime cycle

    // Lowering warnings
    pub const W0001: DiagnosticCode = DiagnosticCode("W0001"); // Missing parameter type annotation
    pub const W0002: DiagnosticCode = DiagnosticCode("W0002"); // Missing return type annotation
}

/// Style for rendering diagnostic annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationStyle {
    /// Primary error location (red underline)
    Primary,
    /// Secondary related location (blue underline)
    Secondary,
    /// Information location (no underline, just marker)
    Info,
}

/// Single annotation on a source location
#[derive(Debug, Clone)]
pub struct Annotation {
    /// Source span to annotate
    pub span: Span,
    /// Style of annotation
    pub style: AnnotationStyle,
    /// Message for this annotation
    pub message: Option<String>,
}

impl Annotation {
    pub fn primary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            style: AnnotationStyle::Primary,
            message: Some(message.into()),
        }
    }

    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            style: AnnotationStyle::Secondary,
            message: Some(message.into()),
        }
    }

    pub fn info(span: Span) -> Self {
        Self {
            span,
            style: AnnotationStyle::Info,
            message: None,
        }
    }
}

/// Suggestion for fixing a diagnostic
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Span to replace
    pub span: Span,
    /// Replacement text
    pub replacement: String,
    /// Description of the suggestion
    pub message: String,
    /// Whether this is a machine-applicable suggestion
    pub applicability: SuggestionApplicability,
}

/// How applicable a suggestion is
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionApplicability {
    /// Safe to apply automatically
    MachineApplicable,
    /// May have false positives or negatives
    MaybeIncorrect,
    /// Probably not what the user wants
    HasPlaceholders,
    /// Only for demonstration purposes
    Unspecified,
}

/// Complete diagnostic message
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub level: DiagnosticLevel,
    /// Error code (optional)
    pub code: Option<DiagnosticCode>,
    /// Primary message
    pub message: String,
    /// Source annotations
    pub annotations: Vec<Annotation>,
    /// Help messages
    pub help: Vec<String>,
    /// Note messages
    pub notes: Vec<String>,
    /// Suggestions for fixes
    pub suggestions: Vec<Suggestion>,
}

impl Diagnostic {
    /// Create a new diagnostic
    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            code: None,
            message: message.into(),
            annotations: Vec::new(),
            help: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Create an error diagnostic
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Error, message)
    }

    /// Create a warning diagnostic
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(DiagnosticLevel::Warning, message)
    }

    /// Add error code
    pub fn with_code(mut self, code: DiagnosticCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Add primary annotation
    pub fn with_primary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.annotations.push(Annotation::primary(span, message));
        self
    }

    /// Add secondary annotation
    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.annotations.push(Annotation::secondary(span, message));
        self
    }

    /// Add help message
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// Add note message
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add suggestion
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add simple suggestion
    pub fn suggest(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        self.suggestions.push(Suggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
            applicability: SuggestionApplicability::MachineApplicable,
        });
        self
    }
}

/// Diagnostic display formatting trait
pub trait DiagnosticDisplay {
    /// Format the diagnostic for display
    fn fmt_diagnostic(
        &self,
        diagnostic: &Diagnostic,
        source_map: &SourceMap,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

/// Default console-based diagnostic formatter (similar to rustc)
pub struct ConsoleDiagnosticDisplay {
    /// Whether to use colors
    pub use_colors: bool,
    /// Maximum number of lines to show in context
    pub context_lines: usize,
}

impl Default for ConsoleDiagnosticDisplay {
    fn default() -> Self {
        Self {
            use_colors: true,
            context_lines: 3,
        }
    }
}

impl ConsoleDiagnosticDisplay {
    /// Format level with optional color
    fn format_level(&self, level: DiagnosticLevel) -> String {
        if self.use_colors {
            match level {
                DiagnosticLevel::Ice => format!("\x1b[1;91m{}\x1b[0m", level), // Bright red
                DiagnosticLevel::Error => format!("\x1b[1;31m{}\x1b[0m", level), // Red
                DiagnosticLevel::Warning => format!("\x1b[1;33m{}\x1b[0m", level), // Yellow
                DiagnosticLevel::Note => format!("\x1b[1;36m{}\x1b[0m", level), // Cyan
                DiagnosticLevel::Help => format!("\x1b[1;32m{}\x1b[0m", level), // Green
            }
        } else {
            format!("{}", level)
        }
    }

    /// Format error code
    fn format_code(&self, code: DiagnosticCode) -> String {
        if self.use_colors {
            format!("\x1b[1m[{}]\x1b[0m", code)
        } else {
            format!("[{}]", code)
        }
    }

    /// Format annotation underline
    fn format_underline(&self, style: AnnotationStyle, length: usize) -> String {
        let char = match style {
            AnnotationStyle::Primary => '^',
            AnnotationStyle::Secondary => '-',
            AnnotationStyle::Info => '-',
        };

        let underline = char.to_string().repeat(length.max(1));

        if self.use_colors {
            match style {
                AnnotationStyle::Primary => format!("\x1b[1;31m{}\x1b[0m", underline), // Red
                AnnotationStyle::Secondary => format!("\x1b[1;34m{}\x1b[0m", underline), // Blue
                AnnotationStyle::Info => format!("\x1b[1;37m{}\x1b[0m", underline),    // White
            }
        } else {
            underline
        }
    }
}

impl DiagnosticDisplay for ConsoleDiagnosticDisplay {
    fn fmt_diagnostic(
        &self,
        diagnostic: &Diagnostic,
        source_map: &SourceMap,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Main diagnostic header
        let level_str = self.format_level(diagnostic.level);
        let code_str = diagnostic
            .code
            .map(|c| format!("{} ", self.format_code(c)))
            .unwrap_or_default();

        writeln!(f, "{}{}: {}", code_str, level_str, diagnostic.message)?;

        // Group annotations by file and sort by line
        let mut file_annotations: HashMap<String, Vec<&Annotation>> = HashMap::new();

        // Get the first source file name from the source map (for single-file programs)
        // TODO: For multi-file programs, track file_id in Span
        let default_filename = source_map
            .get_file_by_id(0)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| "input.zy".to_string());

        for annotation in &diagnostic.annotations {
            file_annotations
                .entry(default_filename.clone())
                .or_default()
                .push(annotation);
        }

        // Display each file's annotations
        for (filename, annotations) in file_annotations {
            // Get the source file
            let source_file = source_map.get_file(&filename);

            // Sort annotations by span start
            let mut sorted_annotations = annotations;
            sorted_annotations.sort_by_key(|a| a.span.start);

            // Display source lines with annotations
            for annotation in sorted_annotations {
                // Get the actual location and source line
                if let Some(file) = source_file {
                    let location = file.get_location(annotation.span.start);
                    let source_line = file.get_line(location.line).unwrap_or("");

                    writeln!(
                        f,
                        "  --> {}:{}:{}",
                        filename, location.line, location.column
                    )?;
                    writeln!(f, "   |")?;
                    writeln!(f, "{:3} | {}", location.line, source_line)?;

                    // Calculate underline position and length
                    let underline_start = location.column - 1; // Convert to 0-based
                    let underline_len = annotation.span.len().max(1);

                    writeln!(
                        f,
                        "   | {}{}",
                        " ".repeat(underline_start),
                        self.format_underline(annotation.style, underline_len)
                    )?;

                    if let Some(message) = &annotation.message {
                        writeln!(f, "   | {}{}", " ".repeat(underline_start), message)?;
                    }
                } else {
                    // Fallback to placeholder if source file not found
                    writeln!(f, "  --> {}:{}:{}", filename, 1, 1)?;
                    writeln!(f, "   |")?;
                    writeln!(f, "{:3} | {}", 1, "    // source code line here")?;
                    writeln!(
                        f,
                        "   | {}{}",
                        " ".repeat(4),
                        self.format_underline(annotation.style, 10)
                    )?;

                    if let Some(message) = &annotation.message {
                        writeln!(f, "   | {}{}", " ".repeat(4), message)?;
                    }
                }
            }

            writeln!(f, "   |")?;
        }

        // Display notes
        for note in &diagnostic.notes {
            writeln!(
                f,
                "   = {}: {}",
                self.format_level(DiagnosticLevel::Note),
                note
            )?;
        }

        // Display help messages
        for help in &diagnostic.help {
            writeln!(
                f,
                "   = {}: {}",
                self.format_level(DiagnosticLevel::Help),
                help
            )?;
        }

        // Display suggestions
        for suggestion in &diagnostic.suggestions {
            writeln!(f, "   = help: {}", suggestion.message)?;
            writeln!(f, "   |")?;
            writeln!(f, "   | {}", suggestion.replacement)?;
        }

        Ok(())
    }
}

// ─── ariadne renderer ──────────────────────────────────────────────────
//
// Phase L: default renderer. `ConsoleDiagnosticDisplay` is the legacy
// hand-rolled ANSI escape path. `AriadneDiagnosticDisplay` routes
// through the `ariadne` crate (snippet boxes, arrows, multi-span
// support, colour fallback). Both implement `DiagnosticDisplay`.
//
// Multi-file caveat: `Span` carries no `file_id` today (see the TODO
// at `ConsoleDiagnosticDisplay::fmt_diagnostic` ~line 358). We
// preserve the single-file assumption — the renderer pulls the
// filename from `SourceMap::get_file_by_id(0)` and labels every
// annotation against that. Threading `file_id` through `Span` is a
// separate piece of work.

/// `ariadne`-backed diagnostic renderer. Default for the compiler
/// from Phase L onward. Produces snippet boxes with arrow annotations
/// and a chevron-style multi-span layout.
pub struct AriadneDiagnosticDisplay {
    /// Whether to use ANSI colour. Default true; falls back gracefully
    /// when the writer is non-tty (ariadne handles that automatically
    /// via its `with_color` config flag).
    pub use_colors: bool,
}

impl Default for AriadneDiagnosticDisplay {
    fn default() -> Self {
        Self { use_colors: true }
    }
}

/// `ariadne::Cache` adapter that pulls source from a `SourceMap`.
/// Cache `Id` is the filename string. ariadne calls `fetch` lazily as
/// it renders, so we build the `ariadne::Source` on first use and
/// memoize it for subsequent labels.
struct SourceMapCache<'a> {
    source_map: &'a SourceMap,
    cached: HashMap<String, ariadne::Source<String>>,
}

impl<'a> ariadne::Cache<String> for SourceMapCache<'a> {
    type Storage = String;

    fn fetch(&mut self, id: &String) -> Result<&ariadne::Source<Self::Storage>, impl fmt::Debug> {
        if !self.cached.contains_key(id) {
            // Explicit binding so both arms of the `?`-shaped path
            // produce a single concrete `String` error type — ariadne's
            // `Result<_, impl Debug>` return type can't otherwise infer
            // it.
            let content: Result<String, String> = self
                .source_map
                .get_file(id)
                .map(|f| f.content.clone())
                .ok_or_else(|| format!("source file not found in map: {id}"));
            let content = content?;
            self.cached
                .insert(id.clone(), ariadne::Source::from(content));
        }
        let result: Result<&ariadne::Source<String>, String> =
            Ok(self.cached.get(id).expect("just inserted"));
        result
    }

    fn display<'b>(&self, id: &'b String) -> Option<impl fmt::Display + 'b> {
        Some(id.clone())
    }
}

impl AriadneDiagnosticDisplay {
    fn level_to_kind(
        level: DiagnosticLevel,
    ) -> (ariadne::ReportKind<'static>, Option<&'static str>) {
        match level {
            DiagnosticLevel::Ice => (ariadne::ReportKind::Error, Some("ICE")),
            DiagnosticLevel::Error => (ariadne::ReportKind::Error, None),
            DiagnosticLevel::Warning => (ariadne::ReportKind::Warning, None),
            DiagnosticLevel::Note => (ariadne::ReportKind::Advice, None),
            DiagnosticLevel::Help => (ariadne::ReportKind::Advice, None),
        }
    }

    fn style_to_color(style: AnnotationStyle) -> ariadne::Color {
        match style {
            AnnotationStyle::Primary => ariadne::Color::Red,
            AnnotationStyle::Secondary => ariadne::Color::Blue,
            // ariadne has no semantic "info" lane; neutral white is
            // close enough and stays readable on dark + light terms.
            AnnotationStyle::Info => ariadne::Color::White,
        }
    }
}

impl DiagnosticDisplay for AriadneDiagnosticDisplay {
    fn fmt_diagnostic(
        &self,
        diagnostic: &Diagnostic,
        source_map: &SourceMap,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Single-file assumption: pull the filename from file_id 0.
        // Multi-file diagnostics would need a `file_id` on `Span` —
        // tracked at the TODO inside `ConsoleDiagnosticDisplay`.
        let filename = source_map
            .get_file_by_id(0)
            .map(|sf| sf.name.clone())
            .unwrap_or_else(|| "input.zy".to_string());

        let (kind, prefix) = Self::level_to_kind(diagnostic.level);
        let header = match prefix {
            Some(p) => format!("{p}: {}", diagnostic.message),
            None => diagnostic.message.clone(),
        };

        // Anchor the report at the primary annotation's span when one
        // exists; fall back to (file, 0) otherwise so even
        // span-less diagnostics still render a header.
        let anchor_start = diagnostic
            .annotations
            .iter()
            .find(|a| a.style == AnnotationStyle::Primary)
            .or_else(|| diagnostic.annotations.first())
            .map(|a| a.span.start)
            .unwrap_or(0);

        let mut builder =
            ariadne::Report::build(kind, (filename.clone(), anchor_start..anchor_start))
                .with_message(header)
                .with_config(ariadne::Config::new().with_color(self.use_colors));

        if let Some(code) = diagnostic.code {
            builder = builder.with_code(code.0);
        }

        for annotation in &diagnostic.annotations {
            // ariadne dislikes zero-length ranges in labels (no
            // visible underline); pad by 1 char so the caret still
            // appears at the right column.
            let start = annotation.span.start;
            let end = if annotation.span.end > annotation.span.start {
                annotation.span.end
            } else {
                annotation.span.start + 1
            };
            let mut label = ariadne::Label::new((filename.clone(), start..end))
                .with_color(Self::style_to_color(annotation.style));
            if let Some(msg) = &annotation.message {
                label = label.with_message(msg.clone());
            }
            builder = builder.with_label(label);
        }

        for note in &diagnostic.notes {
            builder = builder.with_note(note.clone());
        }

        for help in &diagnostic.help {
            builder = builder.with_help(help.clone());
        }

        // ariadne's `Report::write_for_stdout` wants a writer + cache.
        // We render into a Vec<u8>, then push the bytes through the
        // user's `fmt::Formatter`. This is the standard adapter
        // pattern when bridging `io::Write` consumers into a
        // `fmt::Write` sink.
        let mut cache = SourceMapCache {
            source_map,
            cached: HashMap::new(),
        };
        let mut buf: Vec<u8> = Vec::new();
        if builder
            .finish()
            .write_for_stdout(&mut cache, &mut buf)
            .is_err()
        {
            // Renderer failure shouldn't crash the compile; fall
            // back to the message line so the user at least sees
            // what went wrong.
            return writeln!(f, "{}: {}", diagnostic.level, diagnostic.message);
        }
        f.write_str(&String::from_utf8_lossy(&buf))?;

        // Render suggestions separately (ariadne has no native
        // suggestion lane). Match the legacy formatter's style.
        for suggestion in &diagnostic.suggestions {
            writeln!(f, "   = help: {}", suggestion.message)?;
            writeln!(f, "   |")?;
            writeln!(f, "   | {}", suggestion.replacement)?;
        }

        Ok(())
    }
}

/// Diagnostic collector that accumulates diagnostics during compilation
pub struct DiagnosticCollector {
    /// Collected diagnostics
    diagnostics: Vec<Diagnostic>,
    /// Error count
    error_count: usize,
    /// Warning count  
    warning_count: usize,
    /// Whether to stop on first error
    fatal_on_error: bool,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
            fatal_on_error: false,
        }
    }

    /// Set fatal-on-error mode
    pub fn with_fatal_on_error(mut self, fatal: bool) -> Self {
        self.fatal_on_error = fatal;
        self
    }

    /// Add a diagnostic
    pub fn add(&mut self, diagnostic: Diagnostic) -> Result<(), ()> {
        match diagnostic.level {
            DiagnosticLevel::Error | DiagnosticLevel::Ice => {
                self.error_count += 1;
                self.diagnostics.push(diagnostic);
                if self.fatal_on_error {
                    return Err(());
                }
            }
            DiagnosticLevel::Warning => {
                self.warning_count += 1;
                self.diagnostics.push(diagnostic);
            }
            _ => {
                self.diagnostics.push(diagnostic);
            }
        }
        Ok(())
    }

    /// Emit an error
    pub fn error(&mut self, message: impl Into<String>) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Diagnostic::error(message))
    }

    /// Emit a warning
    pub fn warning(&mut self, message: impl Into<String>) -> DiagnosticBuilder<'_> {
        DiagnosticBuilder::new(self, Diagnostic::warning(message))
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }

    /// Display all diagnostics
    pub fn display_all<D: DiagnosticDisplay>(&self, display: &D, source_map: &SourceMap) -> String {
        let mut output = String::new();
        for diagnostic in &self.diagnostics {
            output.push_str(&format!(
                "{}",
                DisplayWrapper {
                    diagnostic,
                    display,
                    source_map
                }
            ));
            output.push('\n');
        }
        output
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing diagnostics fluently
pub struct DiagnosticBuilder<'a> {
    collector: &'a mut DiagnosticCollector,
    diagnostic: Diagnostic,
}

impl<'a> DiagnosticBuilder<'a> {
    fn new(collector: &'a mut DiagnosticCollector, diagnostic: Diagnostic) -> Self {
        Self {
            collector,
            diagnostic,
        }
    }

    /// Add error code
    pub fn code(mut self, code: DiagnosticCode) -> Self {
        self.diagnostic.code = Some(code);
        self
    }

    /// Add primary span
    pub fn primary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic
            .annotations
            .push(Annotation::primary(span, message));
        self
    }

    /// Add secondary span
    pub fn secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic
            .annotations
            .push(Annotation::secondary(span, message));
        self
    }

    /// Add help message
    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.diagnostic.help.push(help.into());
        self
    }

    /// Add note
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.diagnostic.notes.push(note.into());
        self
    }

    /// Add suggestion
    pub fn suggest(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        self.diagnostic.suggestions.push(Suggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
            applicability: SuggestionApplicability::MachineApplicable,
        });
        self
    }

    /// Emit the diagnostic
    pub fn emit(self) -> Result<(), ()> {
        self.collector.add(self.diagnostic)
    }
}

/// Wrapper for displaying diagnostics
struct DisplayWrapper<'a, D> {
    diagnostic: &'a Diagnostic,
    display: &'a D,
    source_map: &'a SourceMap,
}

impl<'a, D: DiagnosticDisplay> fmt::Display for DisplayWrapper<'a, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display
            .fmt_diagnostic(self.diagnostic, self.source_map, f)
    }
}

/// Whether diagnostics should be colourised for this process.
///
/// The usual convention, in the usual order: `NO_COLOR` set to anything
/// non-empty turns colour off, `CLICOLOR_FORCE` set to anything but `0`
/// turns it on regardless, and otherwise it follows whether stderr is a
/// terminal. Piping to a file or a pager, and any captured test
/// harness, therefore get plain text without having to ask.
pub fn colors_enabled() -> bool {
    use std::io::IsTerminal;
    if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
        return false;
    }
    if std::env::var_os("CLICOLOR_FORCE").is_some_and(|v| !v.is_empty() && v != "0") {
        return true;
    }
    std::io::stderr().is_terminal()
}

/// Render one diagnostic against one file, as a snippet box.
///
/// The full machinery wants a [`SourceMap`] and a [`DiagnosticDisplay`],
/// which is more than a caller holding a single file and a single error
/// needs. Front ends that parse one file at a time -- a hot reload,
/// a REPL line, a language server's `didSave` -- reach for this
/// instead of assembling both.
///
/// `use_colors` should be false when the output goes anywhere but a
/// terminal: a log file, a UI label, a test assertion.
pub fn render_diagnostic(
    diagnostic: &Diagnostic,
    filename: &str,
    source: &str,
    use_colors: bool,
) -> String {
    let mut source_map = SourceMap::new();
    source_map.add_file(filename.to_string(), source.to_string());
    format!(
        "{}",
        DisplayWrapper {
            diagnostic,
            display: &AriadneDiagnosticDisplay { use_colors },
            source_map: &source_map,
        }
    )
}

/// Extension trait for integrating diagnostics with compiler phases
pub trait WithDiagnostics {
    /// Run with diagnostic collection
    fn with_diagnostics<F, R>(
        &mut self,
        collector: &mut DiagnosticCollector,
        f: F,
    ) -> Result<R, ()>
    where
        F: FnOnce(&mut Self, &mut DiagnosticCollector) -> Result<R, ()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Span;

    #[test]
    fn test_diagnostic_creation() {
        let diag = Diagnostic::error("Type mismatch")
            .with_code(codes::E0001)
            .with_primary(Span::new(10, 20), "expected i32")
            .with_secondary(Span::new(5, 8), "found string")
            .with_help("Consider casting the value to i32")
            .with_note("This error occurs when types don't match");

        assert_eq!(diag.level, DiagnosticLevel::Error);
        assert_eq!(diag.code, Some(codes::E0001));
        assert_eq!(diag.message, "Type mismatch");
        assert_eq!(diag.annotations.len(), 2);
        assert_eq!(diag.help.len(), 1);
        assert_eq!(diag.notes.len(), 1);
    }

    #[test]
    fn ariadne_renders_primary_annotation_without_panicking() {
        // Smoke test the new default renderer: build a diagnostic that
        // exercises a primary annotation, code, note, and help, then
        // render it. We don't pin the exact string (ariadne's exact
        // output may shift across patch versions) — just assert that
        // rendering returns a non-empty string referencing the user's
        // message and code.
        let mut source_map = SourceMap::new();
        source_map.add_file(
            "test.zy".to_string(),
            "fn main() {\n  let x: i32 = \"hello\";\n}\n".to_string(),
        );

        let diag = Diagnostic::error("type mismatch")
            .with_code(codes::E0001)
            .with_primary(Span::new(28, 35), "expected i32, found string")
            .with_secondary(Span::new(17, 20), "declared as i32 here")
            .with_note("the literal is a string but the binding's type is i32")
            .with_help("convert the literal explicitly with `.parse::<i32>()`");

        let renderer = AriadneDiagnosticDisplay {
            // Force no-color so snapshot text stays stable & comparable.
            use_colors: false,
        };
        let rendered = format!(
            "{}",
            DisplayWrapper {
                diagnostic: &diag,
                display: &renderer,
                source_map: &source_map,
            }
        );

        assert!(!rendered.is_empty(), "ariadne should produce output");
        assert!(
            rendered.contains("type mismatch"),
            "diagnostic message must appear; got: {rendered}"
        );
        assert!(
            rendered.contains("E0001"),
            "error code must appear; got: {rendered}"
        );
        assert!(
            rendered.contains("test.zy"),
            "filename must appear; got: {rendered}"
        );
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagnosticCollector::new();

        collector
            .error("First error")
            .code(codes::E0001)
            .primary(Span::new(0, 5), "here")
            .emit()
            .unwrap();

        collector
            .warning("A warning")
            .primary(Span::new(10, 15), "warning here")
            .emit()
            .unwrap();

        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 1);
        assert!(collector.has_errors());
        assert_eq!(collector.diagnostics().len(), 2);
    }
}
