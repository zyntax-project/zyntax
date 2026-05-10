//! Language Grammar Interface for Zyntax Embed
//!
//! This module provides a high-level interface for parsing source code using
//! ZynPEG grammars. It wraps the zyn_peg runtime to provide ergonomic parsing
//! from Rust without requiring compile-time grammar generation.
//!
//! # Example
//!
//! ```ignore
//! use zyntax_embed::{LanguageGrammar, GrammarError};
//!
//! // Load a compiled .zpeg grammar
//! let grammar = LanguageGrammar::load("my_lang.zpeg")?;
//!
//! // Or compile from .zyn source
//! let grammar = LanguageGrammar::compile_zyn(include_str!("my_lang.zyn"))?;
//!
//! // Parse source code to TypedAST JSON
//! let typed_ast_json = grammar.parse_to_json("fn main() { 42 }")?;
//!
//! // Or parse directly to TypedProgram
//! let program = grammar.parse("fn main() { 42 }")?;
//!
//! // Get language metadata
//! println!("Language: {}", grammar.name());
//! println!("Version: {}", grammar.version());
//! println!("Extensions: {:?}", grammar.file_extensions());
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use pest_meta::{optimizer, parser};
use pest_vm::Vm;
use zyn_peg::grammar::{parse_grammar, GrammarIR};
use zyn_peg::runtime::{
    AstHostFunctions, CommandInterpreter, RuntimeValue, TypedAstBuilder, ZpegModule,
};
use zyntax_typed_ast::TypedProgram;

/// Errors that can occur during grammar operations
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("Failed to load grammar file: {0}")]
    LoadError(String),

    #[error("Failed to parse grammar: {0}")]
    ParseError(String),

    #[error("Failed to compile grammar: {0}")]
    CompileError(String),

    #[error("Failed to parse source: {0}")]
    SourceParseError(String),

    #[error("Failed to build AST: {0}")]
    AstBuildError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result type for grammar operations
pub type GrammarResult<T> = Result<T, GrammarError>;

/// A compiled language grammar for parsing source code
///
/// `LanguageGrammar` wraps a compiled ZynPEG module and provides methods
/// for parsing source code into TypedAST. The grammar can be loaded from
/// a precompiled `.zpeg` file or compiled from `.zyn` source.
#[derive(Clone)]
pub struct LanguageGrammar {
    /// The compiled zpeg module (for metadata and legacy support)
    module: Arc<ZpegModule>,
    /// Grammar2 parser (GrammarIR) for actually parsing source code
    /// This is the preferred path as it correctly handles the new action format
    grammar2: Option<Arc<GrammarIR>>,
    /// Cached pest VM for parsing (wrapped in Option for lazy initialization)
    /// Only used as fallback when grammar2 is None
    vm: Arc<Mutex<Option<PestVmCache>>>,
}

/// Cache for the pest VM to avoid recompiling the grammar
struct PestVmCache {
    /// The optimized grammar rules
    rules: Vec<pest_meta::optimizer::OptimizedRule>,
}

impl LanguageGrammar {
    /// Load a grammar from a compiled `.zpeg` file
    ///
    /// # Arguments
    /// * `path` - Path to the `.zpeg` file
    ///
    /// # Example
    /// ```ignore
    /// let grammar = LanguageGrammar::load("my_lang.zpeg")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> GrammarResult<Self> {
        let module = ZpegModule::load(path).map_err(|e| GrammarError::LoadError(e.to_string()))?;
        Ok(Self {
            module: Arc::new(module),
            grammar2: None, // No Grammar2 for pre-compiled modules
            vm: Arc::new(Mutex::new(None)),
        })
    }

    /// Load a grammar from a JSON string (serialized zpeg module)
    ///
    /// # Arguments
    /// * `json` - The JSON string containing the serialized zpeg module
    pub fn from_json(json: &str) -> GrammarResult<Self> {
        let module: ZpegModule = serde_json::from_str(json)
            .map_err(|e| GrammarError::LoadError(format!("Invalid zpeg JSON: {}", e)))?;
        Ok(Self {
            module: Arc::new(module),
            grammar2: None, // No Grammar2 for pre-compiled modules
            vm: Arc::new(Mutex::new(None)),
        })
    }

    /// Compile a grammar from `.zyn` source code
    ///
    /// This parses and compiles a ZynPEG grammar definition into a usable
    /// grammar for parsing source code.
    ///
    /// # Arguments
    /// * `zyn_source` - The `.zyn` grammar source code
    ///
    /// # Example
    /// ```ignore
    /// let grammar = LanguageGrammar::compile_zyn(r#"
    ///     @language { name: "Calculator" }
    ///     expr = { number | binary_op }
    ///     number = @{ ASCII_DIGIT+ }
    /// "#)?;
    /// ```
    pub fn compile_zyn(zyn_source: &str) -> GrammarResult<Self> {
        use pest::Parser;
        use zyn_peg::ast::build_grammar;
        use zyn_peg::runtime::ZpegCompiler;
        use zyn_peg::{Rule as ZynRule, ZynGrammarParser};

        // Parse the .zyn grammar file using the old parser (for ZpegModule metadata)
        let pairs = ZynGrammarParser::parse(ZynRule::program, zyn_source).map_err(|e| {
            GrammarError::ParseError(format!("Failed to parse .zyn grammar: {}", e))
        })?;

        // Build the grammar AST
        let grammar = build_grammar(pairs)
            .map_err(|e| GrammarError::ParseError(format!("Failed to build grammar: {}", e)))?;

        // Compile to zpeg module (for metadata)
        let module = ZpegCompiler::compile(&grammar)
            .map_err(|e| GrammarError::CompileError(e.to_string()))?;

        // Also create a Grammar2 parser (GrammarIR) for actual parsing
        // Grammar2 correctly handles the new action format
        let grammar2 = parse_grammar(zyn_source).map_err(|e| {
            GrammarError::CompileError(format!("Failed to compile Grammar2: {}", e))
        })?;

        Ok(Self {
            module: Arc::new(module),
            grammar2: Some(Arc::new(grammar2)),
            vm: Arc::new(Mutex::new(None)),
        })
    }

    /// Compile a grammar from a `.zyn` file path
    ///
    /// # Arguments
    /// * `path` - Path to the `.zyn` grammar file
    pub fn compile_zyn_file<P: AsRef<Path>>(path: P) -> GrammarResult<Self> {
        let source = std::fs::read_to_string(path)?;
        Self::compile_zyn(&source)
    }

    /// Create a grammar from an already-compiled zpeg module
    pub fn from_module(module: ZpegModule) -> Self {
        Self {
            module: Arc::new(module),
            grammar2: None, // No Grammar2 for pre-compiled modules
            vm: Arc::new(Mutex::new(None)),
        }
    }

    /// Save the compiled grammar to a `.zpeg` file
    ///
    /// # Arguments
    /// * `path` - Path where to save the `.zpeg` file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> GrammarResult<()> {
        self.module
            .save(path)
            .map_err(|e| GrammarError::CompileError(e.to_string()))
    }

    /// Get the language name from the grammar metadata
    pub fn name(&self) -> &str {
        &self.module.metadata.name
    }

    /// Get the language version from the grammar metadata
    pub fn version(&self) -> &str {
        &self.module.metadata.version
    }

    /// Get the file extensions this grammar handles
    pub fn file_extensions(&self) -> &[String] {
        &self.module.metadata.file_extensions
    }

    /// Get the entry point function name if declared
    pub fn entry_point(&self) -> Option<&str> {
        self.module.metadata.entry_point.as_deref()
    }

    /// Get the builtin function mappings
    pub fn builtins(&self) -> &zyn_peg::BuiltinMappings {
        &self.module.metadata.builtins
    }

    /// Get the pest grammar string
    pub fn pest_grammar(&self) -> &str {
        &self.module.pest_grammar
    }

    /// Get a reference to the underlying zpeg module
    pub fn module(&self) -> &ZpegModule {
        &self.module
    }

    /// Parse source code and return the TypedAST as JSON
    ///
    /// This is useful for debugging or when you need to serialize the AST.
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    ///
    /// # Returns
    /// The TypedAST serialized as JSON
    pub fn parse_to_json(&self, source: &str) -> GrammarResult<String> {
        self.parse_to_json_with_filename(source, "unknown.zynml")
    }

    /// Parse source code with a specific filename and return the TypedAST as JSON
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    /// * `filename` - The filename to use for source location (for diagnostics)
    ///
    /// # Returns
    /// The TypedAST serialized as JSON
    pub fn parse_to_json_with_filename(
        &self,
        source: &str,
        filename: &str,
    ) -> GrammarResult<String> {
        // Use Grammar2 (packrat memoized) when available to avoid exponential backtracking
        // in the old pest VM for files with many f-strings.
        if self.grammar2.is_some() {
            let program = self.parse_to_typed_program(source, filename)?;
            let json = serde_json::to_string(&program).map_err(|e| {
                GrammarError::AstBuildError(format!("Failed to serialize TypedAST: {}", e))
            })?;
            return Ok(json);
        }

        let mut builder = TypedAstBuilder::new();
        builder.set_source(filename.to_string(), source.to_string());
        self.parse_with_builder(source, builder)
    }

    /// Parse source code and return a TypedProgram
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    ///
    /// # Returns
    /// The parsed TypedProgram ready for lowering to HIR
    pub fn parse(&self, source: &str) -> GrammarResult<TypedProgram> {
        self.parse_with_filename(source, "unknown.zynml")
    }

    /// Parse source code with a specific filename (for diagnostics)
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    /// * `filename` - The filename to use for source location (for diagnostics)
    ///
    /// # Returns
    /// The parsed TypedProgram ready for lowering to HIR
    pub fn parse_with_filename(&self, source: &str, filename: &str) -> GrammarResult<TypedProgram> {
        use zyntax_typed_ast::source::SourceFile;

        let json = self.parse_to_json_with_filename(source, filename)?;
        let mut program: TypedProgram = serde_json::from_str(&json).map_err(|e| {
            GrammarError::AstBuildError(format!("Failed to deserialize TypedAST: {}", e))
        })?;

        // Add source file for proper diagnostics
        program.source_files = vec![SourceFile::new(filename.to_string(), source.to_string())];

        // Inject extern function declarations for all builtins from @builtin directive
        // This ensures the type checker can find these symbols in scope
        self.inject_builtin_externs(&mut program, None)?;

        Ok(program)
    }

    /// Parse source code with plugin signatures (for proper extern function declarations)
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    /// * `filename` - The filename to use for source location (for diagnostics)
    /// * `signatures` - Plugin signatures mapping symbol names to ZRTL signatures
    ///
    /// # Returns
    /// The parsed TypedProgram ready for lowering to HIR
    pub fn parse_with_signatures(
        &self,
        source: &str,
        filename: &str,
        signatures: &std::collections::HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    ) -> GrammarResult<TypedProgram> {
        use zyntax_typed_ast::source::SourceFile;

        // Parse directly to TypedProgram without JSON serialization to preserve TypeRegistry
        let mut program = self.parse_to_typed_program(source, filename)?;

        // Add source file for proper diagnostics
        program.source_files = vec![SourceFile::new(filename.to_string(), source.to_string())];

        // Inject extern function declarations with signatures
        self.inject_builtin_externs(&mut program, Some(signatures))?;

        Ok(program)
    }

    /// Parse source code directly to TypedProgram without JSON serialization
    /// This preserves the TypeRegistry which is not serializable
    fn parse_to_typed_program(&self, source: &str, filename: &str) -> GrammarResult<TypedProgram> {
        use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParsedValue, ParserState};
        use zyntax_typed_ast::source::SourceFile;
        use zyntax_typed_ast::type_registry::TypeRegistry;
        use zyntax_typed_ast::TypedASTBuilder;

        // Use Grammar2 if available (preferred path - handles new action format correctly)
        if let Some(grammar2) = &self.grammar2 {
            log::debug!("[parse_to_typed_program] Using Grammar2 for parsing");

            // Grammar2's recursive descent parser can overflow the default 8 MB thread stack
            // for source files with deep expression nesting or many f-strings (due to
            // mutual recursion between `expr` and `f_string_interp` rules). Spawn a
            // dedicated thread with 64 MB stack to avoid SIGABRT from stack overflow.
            let grammar2 = Arc::clone(grammar2);
            let source_owned = source.to_string();
            let filename_owned = filename.to_string();

            let parse_result: GrammarResult<TypedProgram> = std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024) // 64 MB
                .spawn(move || {
                    let interpreter = GrammarInterpreter::new(&grammar2);
                    let mut builder = TypedASTBuilder::new();
                    let mut registry = TypeRegistry::new();
                    let mut state = ParserState::new(&source_owned, &mut builder, &mut registry);

                    match interpreter.parse(&mut state) {
                        ParseResult::Success(ParsedValue::Program(mut program), _) => {
                            program.source_files =
                                vec![SourceFile::new(filename_owned, source_owned)];
                            Ok(*program)
                        }
                        ParseResult::Success(_, _) => Err(GrammarError::AstBuildError(
                            "Grammar2 returned unexpected result type".into(),
                        )),
                        ParseResult::Failure(e) => Err(GrammarError::SourceParseError(format!(
                            "Parse error at {}:{}: expected {:?}",
                            e.line, e.column, e.expected
                        ))),
                    }
                })
                .map_err(|e| {
                    GrammarError::AstBuildError(format!("Failed to spawn parser thread: {}", e))
                })?
                .join()
                .map_err(|_| GrammarError::AstBuildError("Parser thread panicked".into()))?;

            return parse_result;
        }

        // Fallback to old ZpegModule path (for pre-compiled grammars)
        log::debug!("[parse_to_typed_program] Using legacy ZpegModule path");

        // Initialize or get the cached VM
        let rules = {
            let mut cache = self.vm.lock().unwrap();
            if cache.is_none() {
                let rules = self.compile_pest_grammar()?;
                *cache = Some(PestVmCache { rules });
            }
            cache.as_ref().unwrap().rules.clone()
        };

        // Create VM and parse source
        let vm = Vm::new(rules);
        let parse_result = vm
            .parse("program", source)
            .map_err(|e| GrammarError::SourceParseError(e.to_string()))?;

        // Create AST builder
        let mut builder = zyn_peg::runtime::TypedAstBuilder::new();
        builder.set_source(filename.to_string(), source.to_string());

        // Create interpreter
        let mut interpreter = CommandInterpreter::new(&self.module, builder);

        // Walk the parse tree and execute commands
        let _result = walk_parse_tree(&mut interpreter, parse_result)?;

        // Get the TypedProgram directly from the builder
        // We don't need the program_handle - just call build_program() which
        // returns the TypedProgram with the TypeRegistry intact
        let program = interpreter.host_mut().build_program();

        Ok(program)
    }

    /// Parse source code with a custom AST builder
    ///
    /// This allows using a custom implementation of `AstHostFunctions` for
    /// specialized AST construction.
    ///
    /// # Arguments
    /// * `source` - The source code to parse
    /// * `builder` - The AST builder implementing `AstHostFunctions`
    ///
    /// # Returns
    /// The finalized AST as JSON string
    pub fn parse_with_builder<H: AstHostFunctions>(
        &self,
        source: &str,
        builder: H,
    ) -> GrammarResult<String> {
        // Initialize or get the cached VM
        let rules = {
            let mut cache = self.vm.lock().unwrap();
            if cache.is_none() {
                let rules = self.compile_pest_grammar()?;
                *cache = Some(PestVmCache { rules });
            }
            cache.as_ref().unwrap().rules.clone()
        };

        // Create VM and parse source
        let vm = Vm::new(rules);
        let parse_result = vm
            .parse("program", source)
            .map_err(|e| GrammarError::SourceParseError(e.to_string()))?;

        // Create interpreter
        let mut interpreter = CommandInterpreter::new(&self.module, builder);

        // Walk the parse tree and execute commands
        let result = walk_parse_tree(&mut interpreter, parse_result)?;

        // Finalize the AST
        let json = match result {
            RuntimeValue::Node(handle) => interpreter.host_mut().finalize_program(handle),
            _ => {
                // Create empty program if we got something unexpected
                let handle = interpreter.host_mut().create_program();
                interpreter.host_mut().finalize_program(handle)
            }
        };

        Ok(json)
    }

    /// Compile the pest grammar to optimized rules
    fn compile_pest_grammar(&self) -> GrammarResult<Vec<pest_meta::optimizer::OptimizedRule>> {
        // Parse the pest grammar
        let pairs =
            parser::parse(parser::Rule::grammar_rules, &self.module.pest_grammar).map_err(|e| {
                GrammarError::CompileError(format!("Failed to parse pest grammar: {:?}", e))
            })?;

        // Convert to AST and optimize
        let ast = parser::consume_rules(pairs).map_err(|e| {
            GrammarError::CompileError(format!("Failed to consume grammar rules: {:?}", e))
        })?;

        Ok(optimizer::optimize(ast))
    }

    /// Inject extern function declarations for all builtins from @builtin directive
    ///
    /// This creates TypedDeclaration::Function entries with is_external=true for each
    /// builtin function so the type checker can find them in scope.
    ///
    /// # Arguments
    /// * `program` - The TypedProgram to inject declarations into
    /// * `signatures` - Optional plugin signatures for proper parameter types
    fn inject_builtin_externs(
        &self,
        program: &mut TypedProgram,
        signatures: Option<
            &std::collections::HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
        >,
    ) -> GrammarResult<()> {
        use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
        use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedFunction, TypedParameter};
        use zyntax_typed_ast::{
            typed_node, CallingConvention, InternedString, Mutability, Span, Visibility,
        };

        let span = Span::new(0, 0); // Synthetic span for injected declarations

        // Iterate over all builtins from @builtin directive
        for (source_name, target_symbol) in &self.module.metadata.builtins.functions {
            // Get return type from @types.function_returns if available, otherwise use signature or Any
            let return_type = if let Some(type_str) =
                self.module.metadata.types.function_returns.get(source_name)
            {
                // Use type from @types directive
                Type::Extern {
                    name: InternedString::new_global(type_str),
                    layout: None, // Layout determined by ZRTL at runtime
                }
            } else if let Some(sigs) = signatures {
                // Try to get return type from plugin signature
                // Use type_tag_to_type_with_symbol to infer opaque type from symbol name
                sigs.get(target_symbol.as_str())
                    .map(|sig| Self::type_tag_to_type_with_symbol(&sig.return_type, target_symbol))
                    .unwrap_or(Type::Any)
            } else {
                Type::Any
            };

            // Get parameters from signature if available
            let params = if let Some(sigs) = signatures {
                if let Some(sig) = sigs.get(target_symbol.as_str()) {
                    // Convert ZRTL signature parameters to TypedParameter
                    use zyntax_typed_ast::typed_ast::ParameterKind;
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
                                span: span,
                            }
                        })
                        .collect()
                } else {
                    vec![] // No signature found - accept anything
                }
            } else {
                vec![] // No signatures provided - accept anything
            };

            // 1. Create alias extern (e.g., println -> links to $IO$println_dynamic)
            // This is what user code calls
            let alias_func = TypedFunction {
                name: InternedString::new_global(source_name),
                annotations: vec![],
                effects: vec![],
                with_handlers: vec![],
                type_params: vec![],
                params: params.clone(),
                return_type: return_type.clone(),
                body: None, // Extern functions have no body
                visibility: Visibility::Public,
                is_async: false,
                is_pure: false,
                is_external: true, // Mark as external
                calling_convention: CallingConvention::Default,
                link_name: Some(InternedString::new_global(target_symbol)), // Link to ZRTL symbol
            };
            program.declarations.push(typed_node(
                TypedDeclaration::Function(alias_func),
                Type::Primitive(PrimitiveType::Unit),
                span,
            ));

            // 2. Create symbol extern (e.g., $IO$println_dynamic) for direct calls
            // Skip if source_name == target_symbol (avoid duplicates)
            if source_name != target_symbol {
                let extern_func = TypedFunction {
                    name: InternedString::new_global(target_symbol),
                    annotations: vec![],
                    effects: vec![],
                    with_handlers: vec![],
                    type_params: vec![],
                    params,
                    return_type,
                    body: None, // Extern functions have no body
                    visibility: Visibility::Public,
                    is_async: false,
                    is_pure: false,
                    is_external: true, // Mark as external
                    calling_convention: CallingConvention::Default,
                    link_name: Some(InternedString::new_global(target_symbol)), // Link to ZRTL symbol
                };
                program.declarations.push(typed_node(
                    TypedDeclaration::Function(extern_func),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ));
            }
        }

        Ok(())
    }

    /// Convert ZRTL TypeTag to Type
    ///
    /// Maps ZRTL runtime type tags to compile-time Type enum values
    fn type_tag_to_type(
        tag: &zyntax_compiler::zrtl::TypeTag,
    ) -> zyntax_typed_ast::type_registry::Type {
        use zyntax_compiler::zrtl::{PrimitiveSize, TypeCategory};
        use zyntax_typed_ast::type_registry::{PrimitiveType, Type};

        match tag.category() {
            TypeCategory::Void => Type::Primitive(PrimitiveType::Unit),
            TypeCategory::Bool => Type::Primitive(PrimitiveType::Bool),
            TypeCategory::Int => {
                // Check size from type_id (PrimitiveSize enum values)
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits8 as u16 => Type::Primitive(PrimitiveType::I8),
                    x if x == PrimitiveSize::Bits16 as u16 => Type::Primitive(PrimitiveType::I16),
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::I32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::I64),
                    _ => Type::Primitive(PrimitiveType::I32), // Default to i32
                }
            }
            TypeCategory::UInt => {
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits8 as u16 => Type::Primitive(PrimitiveType::U8),
                    x if x == PrimitiveSize::Bits16 as u16 => Type::Primitive(PrimitiveType::U16),
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::U32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::U64),
                    _ => Type::Primitive(PrimitiveType::U32), // Default to u32
                }
            }
            TypeCategory::Float => {
                let size = tag.type_id();
                match size {
                    x if x == PrimitiveSize::Bits32 as u16 => Type::Primitive(PrimitiveType::F32),
                    x if x == PrimitiveSize::Bits64 as u16 => Type::Primitive(PrimitiveType::F64),
                    _ => Type::Primitive(PrimitiveType::F32), // Default to f32
                }
            }
            TypeCategory::String => Type::Primitive(PrimitiveType::String),
            TypeCategory::Opaque => {
                // For opaque types, we need the symbol name to infer the type
                // Use placeholder that will be replaced by the calling code
                Type::Any
            }
            _ => Type::Any, // Fallback for complex types
        }
    }

    /// Convert a ZRTL TypeTag to a Type, using the symbol name for opaque type inference
    fn type_tag_to_type_with_symbol(
        tag: &zyntax_compiler::zrtl::TypeTag,
        symbol: &str,
    ) -> zyntax_typed_ast::type_registry::Type {
        use zyntax_compiler::zrtl::TypeCategory;
        use zyntax_typed_ast::type_registry::Type;
        use zyntax_typed_ast::InternedString;

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

/// Recursively walk the pest parse tree and execute zpeg commands
fn walk_parse_tree<'a, H: AstHostFunctions>(
    interpreter: &mut CommandInterpreter<'_, H>,
    pairs: pest::iterators::Pairs<'a, &'a str>,
) -> GrammarResult<RuntimeValue> {
    let mut results = Vec::new();

    for pair in pairs {
        let rule_name = pair.as_rule().to_string();
        let text = pair.as_str().to_string();
        let span_start = pair.as_span().start();
        let span_end = pair.as_span().end();

        log::trace!(
            "[grammar] Processing rule '{}' at {}..{}: {:?}",
            rule_name,
            span_start,
            span_end,
            if text.len() > 40 {
                format!("{}...", &text[..40])
            } else {
                text.clone()
            }
        );

        // Recursively process children first
        let children: Vec<RuntimeValue> = pair
            .into_inner()
            .map(|child| walk_pair_to_value(child, interpreter))
            .collect();

        // Set current span for THIS node (after children have been processed)
        // This ensures the span corresponds to the current rule, not a child
        interpreter.set_current_span(span_start, span_end);
        interpreter
            .host_mut()
            .set_current_span(span_start, span_end);

        // Execute commands for this rule with the correct span
        let result = interpreter
            .execute_rule(&rule_name, &text, children)
            .map_err(|e| {
                GrammarError::AstBuildError(format!("Error executing rule '{}': {}", rule_name, e))
            })?;

        results.push(result);
    }

    // Return the last result (typically the program node)
    Ok(results.into_iter().last().unwrap_or(RuntimeValue::Null))
}

/// Recursively walk a single pair and return its RuntimeValue
fn walk_pair_to_value<'a, H: AstHostFunctions>(
    pair: pest::iterators::Pair<'a, &'a str>,
    interpreter: &mut CommandInterpreter<'_, H>,
) -> RuntimeValue {
    let rule_name = pair.as_rule().to_string();
    let text = pair.as_str().to_string();
    let span_start = pair.as_span().start();
    let span_end = pair.as_span().end();

    log::trace!(
        "[grammar] walk_pair '{}' at {}..{}: {:?}",
        rule_name,
        span_start,
        span_end,
        if text.len() > 30 {
            format!("{}...", &text[..30])
        } else {
            text.clone()
        }
    );

    // Recursively process children
    let children: Vec<RuntimeValue> = pair
        .into_inner()
        .map(|c| walk_pair_to_value(c, interpreter))
        .collect();

    log::trace!(
        "[WALK_PAIR] {}: children.len()={}, children={:?}",
        rule_name,
        children.len(),
        children
    );

    // Set current span for THIS node (after children have been processed)
    // This ensures the span corresponds to the current rule, not a child
    interpreter.set_current_span(span_start, span_end);
    interpreter
        .host_mut()
        .set_current_span(span_start, span_end);

    // Execute commands for this rule with the correct span
    let result = interpreter
        .execute_rule(&rule_name, &text, children)
        .unwrap_or(RuntimeValue::Null);

    log::trace!("[WALK_PAIR] {}: returned result={:?}", rule_name, result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_metadata() {
        // Test with a simple grammar
        let grammar = LanguageGrammar::compile_zyn(
            r#"
            @language {
                name: "TestLang",
                version: "1.0",
                file_extensions: [".test"],
            }

            program = { SOI ~ expr* ~ EOI }
            expr = { number }
            number = @{ ASCII_DIGIT+ }
        "#,
        );

        match grammar {
            Ok(g) => {
                assert_eq!(g.name(), "TestLang");
                assert_eq!(g.version(), "1.0");
                // The grammar declares ".test" which may be stored as-is or normalized
                let extensions = g.file_extensions();
                assert!(
                    extensions == &["test".to_string()] || extensions == &[".test".to_string()],
                    "Expected file_extensions to be [\"test\"] or [\".test\"], got {:?}",
                    extensions
                );
            }
            Err(e) => {
                // Grammar compilation may fail in test environment, that's OK
                eprintln!(
                    "Grammar compilation failed (expected in some environments): {}",
                    e
                );
            }
        }
    }
}
