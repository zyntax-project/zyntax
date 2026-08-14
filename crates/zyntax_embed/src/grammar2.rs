//! Grammar V2 - Direct TypedAST generation using GrammarInterpreter
//!
//! This module provides a simpler, more direct parsing interface that uses
//! ZynPEG 2.0's GrammarInterpreter to parse source code directly into TypedAST
//! without going through JSON serialization or pest VM.
//!
//! # Example
//!
//! ```ignore
//! use zyntax_embed::Grammar2;
//!
//! // Compile from .zyn grammar source
//! let grammar = Grammar2::from_source(include_str!("my_lang.zyn"))?;
//!
//! // Parse source code directly to TypedProgram
//! let program = grammar.parse("fn main() { 42 }")?;
//! ```

use std::sync::Arc;
use zyn_peg::grammar::{
    parse_grammar, BuiltinMappings, GrammarIR, GrammarMetadata, TypeDeclarations,
};
use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParsedValue, ParserState};
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, TypeRegistry};
use zyntax_typed_ast::{
    typed_node, CallingConvention, InternedString, Mutability, Span, Visibility,
};
use zyntax_typed_ast::{
    TypedASTBuilder, TypedDeclaration, TypedFunction, TypedParameter, TypedProgram,
};

/// Errors that can occur during grammar operations
#[derive(Debug, thiserror::Error)]
pub enum Grammar2Error {
    #[error("Failed to parse grammar: {0}")]
    ParseError(String),

    #[error("Failed to parse source: {0}")]
    SourceParseError(String),

    /// A parse failure with the source it came from, so a front end can
    /// render it as a snippet instead of a coordinate pair.
    ///
    /// `Display` still produces the one-line form every existing caller
    /// logs. [`Grammar2Error::render`] produces the snippet.
    #[error("{file}:{line}:{column}: expected {}", expected.join(" or "))]
    SourceParse {
        file: String,
        source_text: String,
        /// Byte offset of the failure, for the annotation span.
        offset: usize,
        line: usize,
        column: usize,
        expected: Vec<String>,
    },

    #[error("Unexpected parse result: expected TypedProgram")]
    UnexpectedResult,
}

/// How many alternatives to name before giving up on the list.
///
/// A PEG at a failed position has tried every branch that could start
/// there, and a grammar with an expression precedence chain offers
/// dozens. Past a handful the list stops being a hint and becomes a
/// wall.
const MAX_EXPECTED_SHOWN: usize = 6;

/// Turn the parser's raw expectation list into something a user can act
/// on: drop the bookkeeping, dedupe, and cap the length.
fn readable_expectations(expected: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    expected
        .iter()
        // Internal bookkeeping, not grammar the user wrote: a packrat
        // cache hit and a repetition-count failure describe how the
        // parser works, not what belongs here.
        .filter(|e| {
            !e.starts_with("memoized")
                && !e.starts_with("expected at least")
                && e.as_str() != "end of input"
                && e.as_str() != "any character"
        })
        .filter(|e| seen.insert(e.as_str()))
        .take(MAX_EXPECTED_SHOWN)
        .cloned()
        .collect()
}

impl Grammar2Error {
    /// [`Grammar2Error::render`], colourised when the process is
    /// attached to a terminal. See
    /// [`zyntax_typed_ast::diagnostics::colors_enabled`] for the rules.
    ///
    /// This is what a CLI or a dev loop wants. Reach for `render(false)`
    /// when the string goes somewhere that can't interpret escapes: a
    /// log file, a UI label, a test assertion.
    pub fn render_auto(&self) -> Option<String> {
        self.render(zyntax_typed_ast::diagnostics::colors_enabled())
    }

    /// Render as a source snippet with the failure underlined, or
    /// `None` for errors that carry no source to point at.
    ///
    /// The result is a block and begins on its own line, so appending
    /// it to a log line ("... error={e}") puts the box under that line
    /// rather than starting it halfway along.
    ///
    /// `use_colors` should be false for anything but a terminal.
    pub fn render(&self, use_colors: bool) -> Option<String> {
        let Self::SourceParse {
            file,
            source_text,
            offset,
            expected,
            ..
        } = self
        else {
            return None;
        };

        // Running out of input is its own failure: the caret has
        // nothing to sit under, so anchor it on the last thing the
        // parser did read and say what happened.
        let trailing_only_space = source_text[*offset..].trim().is_empty();
        let (start, end, at_eof) = if trailing_only_space {
            let last = source_text.trim_end().len();
            (last.saturating_sub(1), last, true)
        } else {
            // The failure is a point, not a range. Underline the token
            // that starts there so the caret has something under it.
            let token_end = source_text[*offset..]
                .find(|c: char| c.is_whitespace())
                .map(|n| offset + n.max(1))
                .unwrap_or(source_text.len())
                .min(source_text.len());
            (*offset, token_end, false)
        };

        let wanted = readable_expectations(expected);
        let (message, label) = match (at_eof, wanted.is_empty()) {
            (true, _) => ("unexpected end of input", "input ended here".to_string()),
            // Nothing nameable left after filtering: the parser matched
            // no branch at all here. "expected end of input", the
            // outermost rule's expectation, is what it says raw, and
            // that reads as nonsense.
            (false, true) => ("unexpected syntax", "the parser stopped here".to_string()),
            (false, false) => (
                "unexpected syntax",
                format!("expected {}", wanted.join(" or ")),
            ),
        };

        let mut diagnostic = zyntax_typed_ast::diagnostics::Diagnostic::error(message)
            .with_primary(
                Span::new(start, end.max(start + 1).min(source_text.len().max(1))),
                label,
            );
        if at_eof {
            diagnostic = diagnostic
                .with_help("something opened earlier was never closed — check the delimiters");
        } else if wanted.is_empty() {
            diagnostic = diagnostic
                .with_help("nothing from here on matched a known form — check for a typo in a keyword, or a missing delimiter");
        }
        let hidden = expected.len().saturating_sub(wanted.len());
        if hidden > 0 && !wanted.is_empty() {
            diagnostic = diagnostic.with_note(match hidden {
                1 => "1 other alternative was possible here".to_string(),
                n => format!("{n} other alternatives were possible here"),
            });
        }
        let report = zyntax_typed_ast::diagnostics::render_diagnostic(
            &diagnostic,
            file,
            source_text,
            use_colors,
        );
        Some(format!("\n{report}"))
    }
}

/// Result type for grammar operations
pub type Grammar2Result<T> = Result<T, Grammar2Error>;

/// A V2 grammar for parsing source code directly to TypedAST
///
/// Uses ZynPEG 2.0's GrammarInterpreter for direct TypedAST construction
/// without JSON serialization or pest VM overhead.
pub struct Grammar2 {
    /// The parsed grammar IR
    grammar: Arc<GrammarIR>,
}

impl Grammar2 {
    pub(crate) fn from_shared_ir(grammar: Arc<GrammarIR>) -> Self {
        Self { grammar }
    }

    /// Create a grammar from .zyn source code
    pub fn from_source(zyn_source: &str) -> Grammar2Result<Self> {
        let grammar =
            parse_grammar(zyn_source).map_err(|e| Grammar2Error::ParseError(e.to_string()))?;

        Ok(Self {
            grammar: Arc::new(grammar),
        })
    }

    /// Create a grammar from an existing GrammarIR
    pub fn from_ir(grammar: GrammarIR) -> Self {
        Self {
            grammar: Arc::new(grammar),
        }
    }

    /// Get the language name
    pub fn name(&self) -> &str {
        &self.grammar.metadata.name
    }

    /// Get the language version
    pub fn version(&self) -> &str {
        &self.grammar.metadata.version
    }

    /// Get the file extensions this grammar handles
    pub fn file_extensions(&self) -> &[String] {
        &self.grammar.metadata.file_extensions
    }

    /// Get the entry point function name if declared
    pub fn entry_point(&self) -> Option<&str> {
        self.grammar.metadata.entry_point.as_deref()
    }

    /// Get a reference to the GrammarIR
    pub fn grammar_ir(&self) -> &GrammarIR {
        &self.grammar
    }

    /// Parse source code directly to TypedProgram
    ///
    /// Uses GrammarInterpreter to parse the source and construct TypedAST nodes.
    /// This bypasses JSON serialization and is the recommended way to parse.
    pub fn parse(&self, source: &str) -> Grammar2Result<TypedProgram> {
        self.parse_with_filename(source, "input.imgpipe")
    }

    /// Parse source code with a specific filename (for diagnostics)
    pub fn parse_with_filename(
        &self,
        source: &str,
        filename: &str,
    ) -> Grammar2Result<TypedProgram> {
        use zyntax_typed_ast::source::SourceFile;

        let grammar = Arc::clone(&self.grammar);
        let source_owned = source.to_string();
        let filename_owned = filename.to_string();

        // Native: parse on a worker thread with a 64 MiB stack so deeply
        // recursive grammars (e.g. expression precedence chains) don't
        // smash the default 8 MiB main-thread stack.
        //
        // wasm32: `std::thread::spawn` returns "operation not supported
        // on this platform" — single-threaded environment. The
        // interpreter's recursion fits in the host's stack budget for
        // the demo programs the wasm shim runs, and there's no
        // configurable stack-size knob in browser JS engines anyway.
        // Run inline.
        let parse_inline = move || -> Grammar2Result<TypedProgram> {
            let interpreter = GrammarInterpreter::new(&grammar);

            let mut builder = TypedASTBuilder::new();
            let mut registry = TypeRegistry::new();
            let mut state = ParserState::new(&source_owned, &mut builder, &mut registry);

            let result = interpreter.parse(&mut state);

            match result {
                ParseResult::Success(ParsedValue::Program(mut program), _) => {
                    program.source_files = vec![SourceFile::new(
                        filename_owned.clone(),
                        source_owned.clone(),
                    )];
                    Ok(*program)
                }
                ParseResult::Success(other, _) => {
                    eprintln!(
                        "[Grammar2] Warning: parse returned {:?}, expected Program",
                        std::mem::discriminant(&other)
                    );
                    Err(Grammar2Error::UnexpectedResult)
                }
                ParseResult::Failure(e) => {
                    // The failure that surfaces is the outermost rule
                    // giving up, which lands at the START of the item
                    // it couldn't consume — pointing at `component` or
                    // `view` and reading as "the keyword isn't
                    // recognised" when the real problem is a missing
                    // brace three lines down. The state records how far
                    // any attempt actually got; that is the position
                    // worth showing.
                    let furthest = state.furthest_error();
                    let e = if furthest.pos > e.pos { furthest } else { e };
                    Err(Grammar2Error::SourceParse {
                        file: filename_owned.clone(),
                        source_text: source_owned.clone(),
                        offset: e.pos.min(source_owned.len()),
                        line: e.line,
                        column: e.column,
                        expected: e.expected.clone(),
                    })
                }
            }
        };

        #[cfg(target_arch = "wasm32")]
        {
            parse_inline()
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            const PARSE_STACK_SIZE_BYTES: usize = 64 * 1024 * 1024;
            let handle = std::thread::Builder::new()
                .name("grammar2-parse".to_string())
                .stack_size(PARSE_STACK_SIZE_BYTES)
                .spawn(parse_inline)
                .map_err(|e| {
                    Grammar2Error::SourceParseError(format!("Failed to spawn parser thread: {}", e))
                })?;

            handle.join().map_err(|panic_payload| {
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic payload".to_string()
                };
                Grammar2Error::SourceParseError(format!(
                    "Grammar2 parser thread panicked: {}",
                    panic_msg
                ))
            })?
        }
    }

    /// Parse source code with plugin signatures (for proper extern declarations)
    pub fn parse_with_signatures(
        &self,
        source: &str,
        filename: &str,
        signatures: &std::collections::HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    ) -> Grammar2Result<TypedProgram> {
        let mut program = self.parse_with_filename(source, filename)?;

        // Inject extern function declarations for builtins
        // This adds both alias names (e.g., image_load) and symbol names (e.g., $Image$load)
        self.inject_builtin_externs(&mut program, Some(signatures))?;

        Ok(program)
    }

    /// Inject extern function declarations for all builtins from @builtin directive
    ///
    /// Creates TWO extern declarations for each builtin:
    /// 1. The alias name (e.g., `image_load`) with link_name pointing to symbol
    /// 2. The symbol name (e.g., `$Image$load`) for direct calls
    fn inject_builtin_externs(
        &self,
        program: &mut TypedProgram,
        signatures: Option<
            &std::collections::HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
        >,
    ) -> Grammar2Result<()> {
        use zyntax_typed_ast::typed_ast::ParameterKind;

        let span = Span::new(0, 0); // Synthetic span for injected declarations

        // Iterate over all builtins from @builtin directive
        for (source_name, target_symbol) in &self.grammar.builtins.functions {
            log::debug!(
                "[Grammar2] Processing builtin: {} -> {}",
                source_name,
                target_symbol
            );

            // Get return type from @types.function_returns if available
            let return_type =
                if let Some(type_str) = self.grammar.type_decls.function_returns.get(source_name) {
                    log::debug!(
                        "[Grammar2] Found @types.function_returns for {}: {}",
                        source_name,
                        type_str
                    );
                    Type::Extern {
                        name: InternedString::new_global(type_str),
                        layout: None,
                    }
                } else if let Some(sigs) = signatures {
                    if let Some(sig) = sigs.get(target_symbol.as_str()) {
                        // Use type_tag_to_type_with_symbol to infer opaque type from symbol name
                        Self::type_tag_to_type_with_symbol(&sig.return_type, target_symbol)
                    } else {
                        Type::Any
                    }
                } else {
                    log::debug!("[Grammar2] No signatures provided, using Type::Any");
                    Type::Any
                };

            // Get parameters from signature if available.
            //
            // When the plugin signature is missing (e.g. on a fresh
            // checkout where `plugins/target/zrtl/*.zrtl` hasn't been
            // built yet), skip the alias entirely. A zero-param stub
            // here clashes downstream with the real call site's
            // multi-param signature (`IncompatibleSignature` panic in
            // the Cranelift backend), which manifests as
            // `hello_simple.zynml` and any other Tensor-using example
            // crashing at first call. Skipping leaves the actual
            // dispatch path to discover the real signature from the
            // ZRTL call.
            let sig_for_symbol = signatures.and_then(|s| s.get(target_symbol.as_str()));
            let params: Vec<TypedParameter> = if let Some(sig) = sig_for_symbol {
                (0..sig.param_count)
                    .map(|i| {
                        let ty = Self::type_tag_to_type(&sig.params[i as usize]);
                        TypedParameter {
                            name: InternedString::new_global(&format!("p{}", i)),
                            ty,
                            mutability: Mutability::Immutable,
                            kind: ParameterKind::Regular,
                            default_value: None,
                            attributes: vec![],
                            span,
                        }
                    })
                    .collect()
            } else {
                log::debug!(
                    "[Grammar2] No signature for builtin `{}` -> `{}`; skipping alias decl",
                    source_name,
                    target_symbol
                );
                continue;
            };

            // 1. Create alias extern (e.g., image_load -> links to $Image$load)
            // This is what the grammar's generated AST calls
            let alias_func = TypedFunction {
                name: InternedString::new_global(source_name),
                annotations: vec![],
                effects: vec![],
                with_handlers: vec![],
                type_params: vec![],
                params: params.clone(),
                return_type: return_type.clone(),
                body: None,
                visibility: Visibility::Public,
                is_async: false,
                is_fiber: false,
                is_pure: false,
                is_external: true,
                calling_convention: CallingConvention::Default,
                link_name: Some(InternedString::new_global(target_symbol)),
            };
            program.declarations.push(typed_node(
                TypedDeclaration::Function(alias_func),
                Type::Primitive(PrimitiveType::Unit),
                span,
            ));

            // 2. Create symbol extern (e.g., $Image$load) for direct calls
            // Skip if source_name == target_symbol (avoid duplicates)
            if source_name != target_symbol {
                let symbol_func = TypedFunction {
                    name: InternedString::new_global(target_symbol),
                    annotations: vec![],
                    effects: vec![],
                    with_handlers: vec![],
                    type_params: vec![],
                    params,
                    return_type,
                    body: None,
                    visibility: Visibility::Public,
                    is_async: false,
                    is_fiber: false,
                    is_pure: false,
                    is_external: true,
                    calling_convention: CallingConvention::Default,
                    link_name: Some(InternedString::new_global(target_symbol)),
                };
                program.declarations.push(typed_node(
                    TypedDeclaration::Function(symbol_func),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ));
            }
        }

        Ok(())
    }

    /// Convert ZRTL TypeTag to Type
    fn type_tag_to_type(tag: &zyntax_compiler::zrtl::TypeTag) -> Type {
        use zyntax_compiler::zrtl::{PrimitiveSize, TypeCategory};

        match tag.category() {
            TypeCategory::Void => Type::Primitive(PrimitiveType::Unit),
            TypeCategory::Bool => Type::Primitive(PrimitiveType::Bool),
            TypeCategory::Int => {
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits8 as u16 => Type::Primitive(PrimitiveType::I8),
                    x if x == PrimitiveSize::Bits16 as u16 => Type::Primitive(PrimitiveType::I16),
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::I32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::I64),
                    _ => Type::Primitive(PrimitiveType::I32),
                }
            }
            TypeCategory::UInt => {
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits8 as u16 => Type::Primitive(PrimitiveType::U8),
                    x if x == PrimitiveSize::Bits16 as u16 => Type::Primitive(PrimitiveType::U16),
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::U32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::U64),
                    _ => Type::Primitive(PrimitiveType::U32),
                }
            }
            TypeCategory::Float => {
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::F32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::F64),
                    _ => Type::Primitive(PrimitiveType::F32),
                }
            }
            TypeCategory::String => Type::Primitive(PrimitiveType::String),
            TypeCategory::Opaque => Type::Any,
            _ => Type::Any,
        }
    }

    /// Convert a ZRTL TypeTag to a Type, using the symbol name for opaque type inference
    fn type_tag_to_type_with_symbol(tag: &zyntax_compiler::zrtl::TypeTag, symbol: &str) -> Type {
        use zyntax_compiler::zrtl::TypeCategory;

        // For opaque types, infer the type name from the symbol
        // e.g., "$Tensor$add" -> type is "$Tensor"
        if tag.category() == TypeCategory::Opaque {
            // Extract type name from symbol: "$Type$method" -> "$Type"
            if symbol.starts_with('$') {
                if let Some(second_dollar) = symbol[1..].find('$') {
                    let type_name = &symbol[..second_dollar + 1]; // Include the leading $
                    return Type::Extern {
                        name: InternedString::new_global(type_name),
                        layout: None,
                    };
                }
            }
            // Couldn't parse symbol, fall back to Any
            Type::Any
        } else {
            Self::type_tag_to_type(tag)
        }
    }
}

impl Clone for Grammar2 {
    fn clone(&self) -> Self {
        Self {
            grammar: Arc::clone(&self.grammar),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar2_creation() {
        let grammar = Grammar2::from_source(
            r#"
            @language {
                name: "Test",
                version: "1.0",
            }

            program = { SOI ~ EOI }
              -> TypedProgram {
                  declarations: [],
              }
        "#,
        );

        match grammar {
            Ok(g) => {
                assert_eq!(g.name(), "Test");
                assert_eq!(g.version(), "1.0");
            }
            Err(e) => {
                eprintln!("Grammar compilation failed: {}", e);
            }
        }
    }
}
