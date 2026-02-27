//! # ZynML - Machine Learning DSL for Zyntax
//!
//! ZynML is a domain-specific language for machine learning that provides:
//!
//! - **Tensor Operations**: Create, manipulate, and compute with tensors
//! - **Audio Processing**: Load, resample, and extract features from audio
//! - **Text Processing**: Tokenization and text preprocessing
//! - **Vector Search**: Similarity search and embeddings
//! - **Model Loading**: Load pre-trained models in SafeTensors format
//!
//! ## Quick Start
//!
//! ```zynml
//! // Load and preprocess audio
//! let audio = audio_load("speech.wav")
//!     |> resample(16000)
//!     |> to_mono()
//!     |> mel_spectrogram(80, 400, 160)
//!
//! // Create and manipulate tensors
//! let x = tensor([[1.0, 2.0], [3.0, 4.0]])
//! let y = x |> transpose() |> reshape([4])
//!
//! // Vector similarity search
//! let query = tensor([0.1, 0.2, 0.3])
//! let results = index |> search(query, 10)
//! ```
//!
//! ## Architecture
//!
//! ZynML is built on the Zyntax compiler infrastructure:
//!
//! 1. **Grammar** (`ml.zyn`): Defines the ZynML syntax using ZynPEG
//! 2. **Runtime**: Executes ZynML programs using ZRTL plugins
//! 3. **Plugins**: Native SIMD-accelerated implementations
//!
//! ## ZRTL Plugins Used
//!
//! - `zrtl_tensor` - Tensor data structure and operations
//! - `zrtl_audio` - Audio loading and feature extraction
//! - `zrtl_text` - Tokenization and text processing
//! - `zrtl_vector` - Vector search and embeddings
//! - `zrtl_model` - Model loading (SafeTensors)
//! - `zrtl_simd` - SIMD-accelerated operations

use anyhow::{Context, Result};
use std::path::Path;
use thiserror::Error;

// Re-export zyntax_embed types for convenience
pub use zyntax_embed::{FromZyntax, LanguageGrammar, RuntimeEvent, TieredRuntime, ZyntaxRuntime};
// Re-export Grammar2 for direct TypedAST parsing
pub use zyntax_embed::{Grammar2, Grammar2Error, Grammar2Result};

/// ZynML-specific errors
#[derive(Error, Debug)]
pub enum ZynMLError {
    #[error("Failed to compile grammar: {0}")]
    GrammarError(String),

    #[error("Failed to load plugin '{plugin}': {reason}")]
    PluginError { plugin: String, reason: String },

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Type error: {0}")]
    TypeError(String),
}

/// The embedded ZynML grammar
pub const ZYNML_GRAMMAR: &str = include_str!("../ml.zyn");

/// The embedded ZynML standard library - prelude
pub const ZYNML_STDLIB_PRELUDE: &str = include_str!("../stdlib/prelude.zynml");

/// The embedded ZynML standard library - tensor
pub const ZYNML_STDLIB_TENSOR: &str = include_str!("../stdlib/tensor.zynml");

/// Required ZRTL plugins for ZynML
pub const REQUIRED_PLUGINS: &[&str] = &[
    "zrtl_tensor",
    "zrtl_audio",
    "zrtl_text",
    "zrtl_vector",
    "zrtl_model",
    "zrtl_simd",
    "zrtl_io",
    "zrtl_fs",
];

/// Optional ZRTL plugins that enhance functionality
pub const OPTIONAL_PLUGINS: &[&str] = &["zrtl_image", "zrtl_json", "zrtl_http"];

/// Runtime profile used by ZynML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZynMLRuntimeProfile {
    /// Classic single-tier Cranelift runtime.
    Classic,
    /// Tiered runtime tuned for development iterations.
    TieredDevelopment,
    /// Tiered runtime tuned for production.
    TieredProduction,
    /// Tiered runtime with LLVM hot-path tier.
    #[cfg(feature = "llvm-backend")]
    TieredProductionLlvm,
}

/// ZynML runtime configuration
#[derive(Debug, Clone)]
pub struct ZynMLConfig {
    /// Directory containing ZRTL plugins
    pub plugins_dir: String,

    /// Whether to load optional plugins
    pub load_optional: bool,

    /// Enable verbose logging
    pub verbose: bool,

    /// Runtime profile / backend tiering mode.
    pub runtime_profile: ZynMLRuntimeProfile,
}

impl Default for ZynMLConfig {
    fn default() -> Self {
        Self {
            plugins_dir: "plugins/target/zrtl".to_string(),
            load_optional: false,
            verbose: false,
            runtime_profile: ZynMLRuntimeProfile::Classic,
        }
    }
}

enum RuntimeEngine {
    Classic(ZyntaxRuntime),
    Tiered(TieredRuntime),
}

/// ZynML runtime - the main entry point for running ZynML programs
pub struct ZynML {
    runtime: RuntimeEngine,
    config: ZynMLConfig,
    grammar: LanguageGrammar,
    /// Grammar2 for direct TypedAST parsing (optional, used for advanced workflows)
    grammar2: Option<Grammar2>,
}

impl ZynML {
    /// Create a new ZynML runtime with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ZynMLConfig::default())
    }

    /// Create a new ZynML runtime with custom configuration
    pub fn with_config(config: ZynMLConfig) -> Result<Self> {
        // Compile the grammar
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR)
            .map_err(|e| ZynMLError::GrammarError(e.to_string()))?;

        // Create runtime engine according to profile.
        let mut runtime = match config.runtime_profile {
            ZynMLRuntimeProfile::Classic => RuntimeEngine::Classic(
                ZyntaxRuntime::new().context("Failed to create Zyntax runtime")?,
            ),
            ZynMLRuntimeProfile::TieredDevelopment => RuntimeEngine::Tiered(
                TieredRuntime::development().context("Failed to create tiered runtime")?,
            ),
            ZynMLRuntimeProfile::TieredProduction => RuntimeEngine::Tiered(
                TieredRuntime::production().context("Failed to create tiered runtime")?,
            ),
            #[cfg(feature = "llvm-backend")]
            ZynMLRuntimeProfile::TieredProductionLlvm => RuntimeEngine::Tiered(
                TieredRuntime::production_llvm().context("Failed to create tiered LLVM runtime")?,
            ),
        };

        // Register stdlib import resolver in classic runtime mode.
        if let RuntimeEngine::Classic(rt) = &mut runtime {
            rt.add_import_resolver(Box::new(|module_name| match module_name {
                "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
                "tensor" => Ok(Some(ZYNML_STDLIB_TENSOR.to_string())),
                _ => Ok(None), // Not a stdlib module
            }));
        } else if config.verbose {
            log::info!(
                "Tiered runtime profile selected; stdlib import resolvers are not enabled in this mode"
            );
        }

        // Load required plugins BEFORE registering grammar
        // This ensures builtin mappings (e.g., println -> $IO$println_dynamic) can find their targets
        let plugins_path = Path::new(&config.plugins_dir);
        for plugin_name in REQUIRED_PLUGINS {
            let plugin_path = plugins_path.join(format!("{}.zrtl", plugin_name));
            if plugin_path.exists() {
                if config.verbose {
                    log::info!("Loading plugin: {}", plugin_name);
                }
                match &mut runtime {
                    RuntimeEngine::Classic(rt) => rt.load_plugin(&plugin_path),
                    RuntimeEngine::Tiered(rt) => rt.load_plugin(&plugin_path),
                }
                .map_err(|e| ZynMLError::PluginError {
                    plugin: plugin_name.to_string(),
                    reason: e.to_string(),
                })?;
            } else if config.verbose {
                log::warn!("Required plugin not found: {}", plugin_path.display());
            }
        }

        // Load optional plugins if requested
        if config.load_optional {
            for plugin_name in OPTIONAL_PLUGINS {
                let plugin_path = plugins_path.join(format!("{}.zrtl", plugin_name));
                if plugin_path.exists() {
                    if config.verbose {
                        log::info!("Loading optional plugin: {}", plugin_name);
                    }
                    match &mut runtime {
                        RuntimeEngine::Classic(rt) => {
                            let _ = rt.load_plugin(&plugin_path);
                        }
                        RuntimeEngine::Tiered(rt) => {
                            let _ = rt.load_plugin(&plugin_path);
                        }
                    }
                }
            }
        }

        // Register the grammar AFTER loading plugins
        // This ensures builtin mappings can find the target external functions
        match &mut runtime {
            RuntimeEngine::Classic(rt) => rt.register_grammar("zynml", grammar.clone()),
            RuntimeEngine::Tiered(rt) => rt.register_grammar("zynml", grammar.clone()),
        }

        // Optionally compile Grammar2 for direct TypedAST parsing
        let grammar2 = Grammar2::from_source(ZYNML_GRAMMAR).ok();

        Ok(Self {
            runtime,
            config,
            grammar,
            grammar2,
        })
    }

    /// Load and compile a ZynML source file
    pub fn load_file(&mut self, path: &Path) -> Result<Vec<String>> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        self.load_source(&source)
    }

    /// Load and compile ZynML source code
    pub fn load_source(&mut self, source: &str) -> Result<Vec<String>> {
        match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt.load_module("zynml", source),
            RuntimeEngine::Tiered(rt) => rt.load_module("zynml", source),
        }
        .context("Failed to compile ZynML program")
    }

    /// Parse source and return the AST as JSON
    pub fn parse_to_json(&self, source: &str) -> Result<String> {
        self.grammar
            .parse_to_json(source)
            .context("Failed to parse ZynML program")
    }

    /// Parse source directly to TypedProgram using Grammar2
    ///
    /// This is more efficient than `parse_to_json` as it bypasses JSON serialization.
    /// Returns an error if Grammar2 is not available.
    pub fn parse_to_typed_ast(&self, source: &str) -> Result<zyntax_embed::TypedProgram> {
        match &self.grammar2 {
            Some(g2) => g2
                .parse(source)
                .map_err(|e| ZynMLError::ParseError(e.to_string()).into()),
            None => Err(ZynMLError::GrammarError("Grammar2 not available".to_string()).into()),
        }
    }

    /// Parse source directly to TypedProgram with filename for diagnostics
    pub fn parse_to_typed_ast_with_filename(
        &self,
        source: &str,
        filename: &str,
    ) -> Result<zyntax_embed::TypedProgram> {
        match &self.grammar2 {
            Some(g2) => g2
                .parse_with_filename(source, filename)
                .map_err(|e| ZynMLError::ParseError(e.to_string()).into()),
            None => Err(ZynMLError::GrammarError("Grammar2 not available".to_string()).into()),
        }
    }

    /// Check if Grammar2 direct parsing is available
    pub fn has_grammar2(&self) -> bool {
        self.grammar2.is_some()
    }

    /// Get a reference to the Grammar2 instance (if available)
    pub fn grammar2(&self) -> Option<&Grammar2> {
        self.grammar2.as_ref()
    }

    /// Call a function by name with no arguments
    pub fn call(&mut self, name: &str) -> Result<()> {
        match &self.runtime {
            RuntimeEngine::Classic(rt) => rt.call::<()>(name, &[]),
            RuntimeEngine::Tiered(rt) => rt.call::<()>(name, &[]),
        }
        .with_context(|| format!("Failed to call function: {}", name))
    }

    /// Call a function and get a result
    pub fn call_with_result<T: FromZyntax + 'static>(&mut self, name: &str) -> Result<T> {
        match &self.runtime {
            RuntimeEngine::Classic(rt) => rt.call::<T>(name, &[]),
            RuntimeEngine::Tiered(rt) => rt.call::<T>(name, &[]),
        }
        .with_context(|| format!("Failed to call function: {}", name))
    }

    /// Run a ZynML program (calls 'main' if it exists)
    pub fn run(&mut self, source: &str) -> Result<()> {
        let functions = self.load_source(source)?;

        if self.config.verbose {
            log::info!("Compiled functions: {:?}", functions);
        }

        // Try to find and call an entry point
        if functions.contains(&"main".to_string()) {
            self.call("main")
        } else if !functions.is_empty() {
            // Call the first function as entry point
            let entry = &functions[0];
            if self.config.verbose {
                log::info!("No 'main' function, calling '{}'", entry);
            }
            self.call(entry)
        } else {
            Ok(()) // No functions to run
        }
    }

    /// Run a ZynML program from a file
    pub fn run_file(&mut self, path: &Path) -> Result<()> {
        // Load the file with proper filename tracking for diagnostics
        let functions = match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt.load_module_file(path),
            RuntimeEngine::Tiered(rt) => rt.load_module_file(path),
        }
        .with_context(|| format!("Failed to load file: {}", path.display()))?;

        if self.config.verbose {
            println!("Loaded {} functions", functions.len());
        }

        // Try to call main() if it exists
        if functions.iter().any(|f| f == "main") {
            if self.config.verbose {
                println!("Calling main()...");
            }
            self.call("main")
        } else {
            Ok(()) // No main function to run
        }
    }

    /// Get reference to the underlying runtime
    pub fn runtime(&self) -> &ZyntaxRuntime {
        match &self.runtime {
            RuntimeEngine::Classic(rt) => rt,
            RuntimeEngine::Tiered(_) => {
                panic!("runtime() is only available in Classic profile; use tiered_runtime()")
            }
        }
    }

    /// Get mutable reference to the underlying runtime
    pub fn runtime_mut(&mut self) -> &mut ZyntaxRuntime {
        match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt,
            RuntimeEngine::Tiered(_) => {
                panic!(
                    "runtime_mut() is only available in Classic profile; use tiered_runtime_mut()"
                )
            }
        }
    }

    /// Get reference to the underlying tiered runtime (if active).
    pub fn tiered_runtime(&self) -> Option<&TieredRuntime> {
        match &self.runtime {
            RuntimeEngine::Classic(_) => None,
            RuntimeEngine::Tiered(rt) => Some(rt),
        }
    }

    /// Get mutable reference to the underlying tiered runtime (if active).
    pub fn tiered_runtime_mut(&mut self) -> Option<&mut TieredRuntime> {
        match &mut self.runtime {
            RuntimeEngine::Classic(_) => None,
            RuntimeEngine::Tiered(rt) => Some(rt),
        }
    }

    /// Get captured runtime semantic events from the active runtime.
    pub fn runtime_events(&self) -> &[RuntimeEvent] {
        match &self.runtime {
            RuntimeEngine::Classic(rt) => rt.runtime_events(),
            RuntimeEngine::Tiered(rt) => rt.runtime_events(),
        }
    }

    /// Drain captured runtime semantic events from the active runtime.
    pub fn drain_runtime_events(&mut self) -> Vec<RuntimeEvent> {
        match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt.drain_runtime_events(),
            RuntimeEngine::Tiered(rt) => rt.drain_runtime_events(),
        }
    }

    /// Register a runtime event sink callback.
    pub fn set_event_sink<F>(&mut self, sink: F)
    where
        F: Fn(&RuntimeEvent) + Send + Sync + 'static,
    {
        match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt.set_event_sink(sink),
            RuntimeEngine::Tiered(rt) => rt.set_event_sink(sink),
        }
    }

    /// Clear the runtime event sink callback.
    pub fn clear_event_sink(&mut self) {
        match &mut self.runtime {
            RuntimeEngine::Classic(rt) => rt.clear_event_sink(),
            RuntimeEngine::Tiered(rt) => rt.clear_event_sink(),
        }
    }

    /// Check if a plugin is loaded
    pub fn has_plugin(&self, name: &str) -> bool {
        // This would need runtime support to check
        // For now, return true for required plugins
        REQUIRED_PLUGINS.contains(&name) || OPTIONAL_PLUGINS.contains(&name)
    }
}

/// Convenience function to run a ZynML file
pub fn run_file(path: &Path, plugins_dir: &Path, verbose: bool) -> Result<()> {
    let config = ZynMLConfig {
        plugins_dir: plugins_dir.to_string_lossy().to_string(),
        verbose,
        ..Default::default()
    };

    let mut zynml = ZynML::with_config(config)?;
    zynml.run_file(path)
}

/// Convenience function to run ZynML source code
pub fn run_source(source: &str, plugins_dir: &Path, verbose: bool) -> Result<()> {
    let config = ZynMLConfig {
        plugins_dir: plugins_dir.to_string_lossy().to_string(),
        verbose,
        ..Default::default()
    };

    let mut zynml = ZynML::with_config(config)?;
    zynml.run(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_compiles() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR);
        assert!(
            grammar.is_ok(),
            "Grammar should compile: {:?}",
            grammar.err()
        );
    }

    #[test]
    fn test_grammar2_compiles() {
        let grammar = Grammar2::from_source(ZYNML_GRAMMAR);
        assert!(
            grammar.is_ok(),
            "Grammar2 should compile: {:?}",
            grammar.err()
        );
    }

    #[test]
    fn test_grammar2_metadata() {
        let grammar = Grammar2::from_source(ZYNML_GRAMMAR).unwrap();
        // Grammar2 provides metadata about the language
        println!("Language name: {}", grammar.name());
        println!("Language version: {}", grammar.version());
        println!("File extensions: {:?}", grammar.file_extensions());
    }

    #[test]
    fn test_parse_simple_let() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json("let x = 42");
        assert!(
            result.is_ok(),
            "Should parse let statement: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_function() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            fn main() {
                let x = 10
                let y = 20
            }
        "#,
        );
        assert!(result.is_ok(), "Should parse function: {:?}", result.err());
    }

    #[test]
    fn test_parse_pipe_operator() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            let result = x |> f(1) |> g()
        "#,
        );
        assert!(
            result.is_ok(),
            "Should parse pipe operator: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_tensor_literal() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            let t = tensor([1.0, 2.0, 3.0])
        "#,
        );
        assert!(
            result.is_ok(),
            "Should parse tensor literal: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_array_literal() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            let arr = [[1, 2], [3, 4]]
        "#,
        );
        assert!(
            result.is_ok(),
            "Should parse array literal: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_ml_pipeline() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            fn process_audio() {
                let audio = audio_load("test.wav")
                let mono = audio |> to_mono()
                let mel = mono |> mel_spectrogram(80, 400, 160)
            }
        "#,
        );
        assert!(
            result.is_ok(),
            "Should parse ML pipeline: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_control_flow() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();
        let result = grammar.parse_to_json(
            r#"
            fn test() {
                if x > 0 {
                    let y = x * 2
                } else {
                    let y = 0
                }

                while i < 10 {
                    i = i + 1
                }
            }
        "#,
        );
        assert!(
            result.is_ok(),
            "Should parse control flow: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_runtime_with_plugins() {
        use std::path::Path;

        // Find plugins directory relative to workspace root
        let plugins_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .unwrap()
            .parent() // workspace root
            .unwrap()
            .join("plugins/target/zrtl");

        if !plugins_dir.exists() {
            eprintln!(
                "Skipping runtime test: plugins not built at {:?}",
                plugins_dir
            );
            return;
        }

        let config = ZynMLConfig {
            plugins_dir: plugins_dir.to_string_lossy().to_string(),
            load_optional: false,
            verbose: true,
            runtime_profile: ZynMLRuntimeProfile::Classic,
        };

        // Create runtime (this loads plugins and registers grammar)
        let runtime_result = ZynML::with_config(config);
        assert!(
            runtime_result.is_ok(),
            "Should create runtime: {:?}",
            runtime_result.err()
        );

        let zynml = runtime_result.unwrap();

        // Verify plugins are recognized
        assert!(zynml.has_plugin("zrtl_tensor"), "Should have tensor plugin");
        assert!(zynml.has_plugin("zrtl_audio"), "Should have audio plugin");
        assert!(zynml.has_plugin("zrtl_vector"), "Should have vector plugin");
    }

    #[test]
    fn test_parse_minimal_function() {
        // Enable logging for this test
        let _ = env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init();

        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();

        // Print the generated pest grammar for debugging
        println!("=== Generated Pest Grammar ===");
        let pest_grammar = grammar.pest_grammar();
        // Print block-related rules
        for line in pest_grammar.lines() {
            if line.contains("block") || line.contains("statement") || line.contains("fn_def") {
                println!("{}", line);
            }
        }
        println!("=== End relevant pest grammar ===\n");

        // Parse the simplest possible function
        let source = r#"fn main() { let x = 42 }"#;

        let result = grammar.parse_to_json(source);
        assert!(
            result.is_ok(),
            "Should parse minimal function: {:?}",
            result.err()
        );

        // Pretty-print the JSON AST for debugging
        let json = result.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let pretty = serde_json::to_string_pretty(&parsed).unwrap();
        println!("=== Minimal Function AST ===");
        println!("{}", pretty);
    }

    #[test]
    fn test_parse_hello_example() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();

        // Parse the hello.zynml example
        let source = r#"
            fn main() {
                println("Hello from ZynML!")

                let x = tensor([1.0, 2.0, 3.0, 4.0])
                let y = tensor([5.0, 6.0, 7.0, 8.0])

                let dot = vec_dot(x, y)
                println("Dot product: ")

                let matrix = tensor([[1.0, 2.0], [3.0, 4.0]])

                let sum = tensor_sum(matrix)
                let mean = tensor_mean(matrix)

                println("Sum: ")
                println("Mean: ")
                println("Done!")
            }
        "#;

        let result = grammar.parse_to_json(source);
        assert!(
            result.is_ok(),
            "Should parse hello example: {:?}",
            result.err()
        );

        // Pretty-print the JSON AST for debugging
        let json = result.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let pretty = serde_json::to_string_pretty(&parsed).unwrap();
        println!("=== Hello Example AST ===");
        println!("{}", pretty);
    }

    #[test]
    fn test_parse_audio_pipeline_example() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();

        // Parse the audio_pipeline.zynml example
        let source = r#"
            fn main() {
                println("=== ZynML Audio Pipeline ===")

                let audio = audio_load("test.wav")

                let sr = audio_sample_rate(audio)
                let duration = audio_duration(audio)

                println("Sample rate: ")
                println("Duration: ")

                let resampled = audio |> resample(16000)
                let mono = resampled |> to_mono()
                let normalized = mono |> normalize()
                let mel = normalized |> mel_spectrogram(80, 400, 160)

                println("Mel spectrogram extracted")
                println("Ready for ML model input")

                audio_free(audio)
            }
        "#;

        let result = grammar.parse_to_json(source);
        assert!(
            result.is_ok(),
            "Should parse audio pipeline example: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_parse_vector_search_example() {
        let grammar = LanguageGrammar::compile_zyn(ZYNML_GRAMMAR).unwrap();

        // Parse the vector_search.zynml example
        let source = r#"
            fn main() {
                println("=== ZynML Vector Search ===")

                let doc1 = tensor([0.1, 0.2, 0.3, 0.4])
                let doc2 = tensor([0.5, 0.1, 0.2, 0.1])
                let doc3 = tensor([0.2, 0.3, 0.4, 0.5])

                doc1 |> vec_normalize()
                doc2 |> vec_normalize()
                doc3 |> vec_normalize()

                let index = flat_create(4)

                flat_add(index, 1, doc1)
                flat_add(index, 2, doc2)
                flat_add(index, 3, doc3)

                println("Added 3 documents to index")

                let query = tensor([0.15, 0.25, 0.35, 0.45])
                query |> vec_normalize()

                let sim1 = vec_cosine(query, doc1)
                let sim2 = vec_cosine(query, doc2)
                let sim3 = vec_cosine(query, doc3)

                println("Cosine similarities:")
                println("  Doc1: ")
                println("  Doc2: ")
                println("  Doc3: ")

                println("Search complete!")
            }
        "#;

        let result = grammar.parse_to_json(source);
        assert!(
            result.is_ok(),
            "Should parse vector search example: {:?}",
            result.err()
        );
    }
}
