//! # Import Resolution System
//!
//! This module provides an extensible trait-based import resolution system
//! that allows different languages to implement their own import resolution logic.
//!
//! ## Overview
//!
//! Languages like Haxe, Rust, Java, etc. have different module systems:
//! - Haxe: `import haxe.ds.StringMap;`
//! - Rust: `use std::collections::HashMap;`
//! - Java: `import java.util.HashMap;`
//!
//! The `ImportResolver` trait allows each language frontend to provide
//! its own resolution logic while sharing the common TypedAST infrastructure.

use crate::{Span, Type, TypedImport};
use std::collections::HashMap;
use std::path::PathBuf;

/// Result of resolving an import
#[derive(Debug, Clone)]
pub enum ResolvedImport {
    /// Successfully resolved to a type definition
    Type {
        /// The fully qualified name
        qualified_name: Vec<String>,
        /// The resolved type
        ty: Type,
        /// Whether this is an external/runtime type
        is_extern: bool,
    },

    /// Successfully resolved to a module (namespace)
    Module {
        /// The module path
        path: Vec<String>,
        /// Exported symbols from this module
        exports: Vec<ExportedSymbol>,
    },

    /// Resolved to a function
    Function {
        /// The fully qualified function name
        qualified_name: Vec<String>,
        /// Parameter types
        params: Vec<Type>,
        /// Return type
        return_type: Type,
        /// Whether this is an external/runtime function
        is_extern: bool,
    },

    /// Resolved to a constant value
    Constant {
        /// The fully qualified name
        qualified_name: Vec<String>,
        /// The constant's type
        ty: Type,
    },

    /// Glob import - resolved to multiple symbols
    Glob {
        /// The module path
        module_path: Vec<String>,
        /// All exported symbols
        symbols: Vec<ExportedSymbol>,
    },
}

/// A symbol exported from a module
#[derive(Debug, Clone)]
pub struct ExportedSymbol {
    /// The symbol name
    pub name: String,
    /// The symbol kind
    pub kind: SymbolKind,
    /// Whether it's public
    pub is_public: bool,
}

/// Kind of exported symbol
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Type,
    Function,
    Constant,
    Module,
    Class,
    Enum,
    Interface,
    Trait,
}

/// Error during import resolution
#[derive(Debug, Clone)]
pub enum ImportError {
    /// Module not found
    ModuleNotFound {
        path: Vec<String>,
        span: Span,
        suggestions: Vec<String>,
    },

    /// Symbol not found in module
    SymbolNotFound {
        symbol: String,
        module_path: Vec<String>,
        span: Span,
        available: Vec<String>,
    },

    /// Circular import detected
    CircularImport {
        path: Vec<String>,
        cycle: Vec<Vec<String>>,
        span: Span,
    },

    /// Visibility error - symbol is private
    PrivateSymbol {
        symbol: String,
        module_path: Vec<String>,
        span: Span,
    },

    /// Ambiguous import - multiple symbols match
    AmbiguousImport {
        symbol: String,
        candidates: Vec<Vec<String>>,
        span: Span,
    },

    /// File I/O error
    IoError {
        path: PathBuf,
        message: String,
        span: Span,
    },

    /// Parse error in imported module
    ParseError {
        path: PathBuf,
        message: String,
        span: Span,
    },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::ModuleNotFound {
                path, suggestions, ..
            } => {
                write!(f, "module not found: {}", path.join("."))?;
                if !suggestions.is_empty() {
                    write!(f, ", did you mean: {}?", suggestions.join(", "))?;
                }
                Ok(())
            }
            ImportError::SymbolNotFound {
                symbol,
                module_path,
                ..
            } => {
                write!(
                    f,
                    "symbol '{}' not found in module '{}'",
                    symbol,
                    module_path.join(".")
                )
            }
            ImportError::CircularImport { path, .. } => {
                write!(f, "circular import detected: {}", path.join("."))
            }
            ImportError::PrivateSymbol {
                symbol,
                module_path,
                ..
            } => {
                write!(
                    f,
                    "symbol '{}' in '{}' is private",
                    symbol,
                    module_path.join(".")
                )
            }
            ImportError::AmbiguousImport {
                symbol, candidates, ..
            } => {
                write!(
                    f,
                    "ambiguous import '{}', could be: {}",
                    symbol,
                    candidates
                        .iter()
                        .map(|c| c.join("."))
                        .collect::<Vec<_>>()
                        .join(" or ")
                )
            }
            ImportError::IoError { path, message, .. } => {
                write!(f, "I/O error reading '{}': {}", path.display(), message)
            }
            ImportError::ParseError { path, message, .. } => {
                write!(f, "parse error in '{}': {}", path.display(), message)
            }
        }
    }
}

impl std::error::Error for ImportError {}

/// Module file architecture defines how module paths map to filesystem paths
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleArchitecture {
    /// Java/Haxe style: com.example.MyClass -> com/example/MyClass.hx
    /// Package path directly maps to directory structure
    DotSeparatedPackages {
        /// File extension for source files
        extension: String,
    },

    /// Rust style: crate::module::submodule -> module/submodule.rs or module/submodule/mod.rs
    /// Modules can be files or directories with mod.rs
    RustStyle {
        /// File extension for source files
        extension: String,
        /// Name of the module file in directories (e.g., "mod.rs")
        mod_file_name: String,
    },

    /// Python style: package.module -> package/module.py or package/module/__init__.py
    PythonStyle {
        /// File extension for source files
        extension: String,
        /// Name of package init file (e.g., "__init__.py")
        init_file_name: String,
    },

    /// Node.js style: Uses package.json "main" field or index.js
    NodeStyle {
        /// File extensions to try in order (e.g., [".js", ".ts", ".json"])
        extensions: Vec<String>,
        /// Index file name (e.g., "index")
        index_name: String,
    },

    /// Go style: Domain-based import paths
    /// github.com/user/repo/package -> $GOPATH/src/github.com/user/repo/package
    /// or uses go.mod for module resolution
    GoStyle {
        /// Base path for downloaded modules (GOPATH/src or go mod cache)
        module_cache: PathBuf,
        /// File extension
        extension: String,
    },

    /// Deno/URL style: Direct URL imports
    /// https://deno.land/std/http/server.ts
    /// Supports caching and versioning
    UrlStyle {
        /// Local cache directory for downloaded modules
        cache_dir: PathBuf,
        /// Allowed URL schemes (e.g., ["https", "http", "file"])
        allowed_schemes: Vec<String>,
        /// Import map for URL aliasing (maps short names to full URLs)
        import_map: HashMap<String, String>,
    },

    /// Custom architecture with a path resolver function
    Custom {
        /// Name for debugging
        name: String,
    },
}

/// Represents a module source location
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleSource {
    /// Local file path
    FilePath(PathBuf),
    /// Remote URL
    Url(String),
    /// Cached from URL (URL, local cache path)
    CachedUrl { url: String, cache_path: PathBuf },
    /// Virtual/builtin module
    Virtual(String),
    /// Pre-compiled module (MIR/ZBC cache)
    Compiled {
        /// Original source path
        source: Box<ModuleSource>,
        /// Path to compiled cache file (.mir or .zbc)
        cache_path: PathBuf,
        /// Hash of source for cache invalidation
        source_hash: String,
    },
}

/// Cache format for pre-compiled modules
#[derive(Debug, Clone, PartialEq)]
pub enum CompiledCacheFormat {
    /// Zyntax Bytecode - binary representation of MIR (platform-independent)
    /// This is the primary cache format for incremental compilation
    ZBC,
    /// TypedAST JSON (for debugging and interop)
    TypedAstJson,
    /// LLVM IR (for LLVM backend, textual)
    LLVMIR,
    /// LLVM Bitcode (for LLVM backend, binary)
    LLVMBitcode,
    /// Native object file (platform-specific)
    Object,
    /// WebAssembly binary
    Wasm,
}

/// Metadata for cached compiled module
#[derive(Debug, Clone)]
pub struct CompiledModuleCache {
    /// The module path
    pub module_path: Vec<String>,
    /// Original source location
    pub source: ModuleSource,
    /// Cache format
    pub format: CompiledCacheFormat,
    /// Path to cache file
    pub cache_path: PathBuf,
    /// Source file hash (for invalidation)
    pub source_hash: String,
    /// Timestamp of source file when compiled
    pub source_mtime: u64,
    /// Dependencies (other modules this one imports)
    pub dependencies: Vec<Vec<String>>,
    /// Exported symbols
    pub exports: Vec<ExportedSymbol>,
}

impl Default for ModuleArchitecture {
    fn default() -> Self {
        ModuleArchitecture::DotSeparatedPackages {
            extension: "hx".to_string(),
        }
    }
}

impl ModuleArchitecture {
    /// Create a Haxe-style module architecture
    pub fn haxe() -> Self {
        ModuleArchitecture::DotSeparatedPackages {
            extension: "hx".to_string(),
        }
    }

    /// Create a Java-style module architecture
    pub fn java() -> Self {
        ModuleArchitecture::DotSeparatedPackages {
            extension: "java".to_string(),
        }
    }

    /// Create a Rust-style module architecture
    pub fn rust() -> Self {
        ModuleArchitecture::RustStyle {
            extension: "rs".to_string(),
            mod_file_name: "mod.rs".to_string(),
        }
    }

    /// Create a Python-style module architecture
    pub fn python() -> Self {
        ModuleArchitecture::PythonStyle {
            extension: "py".to_string(),
            init_file_name: "__init__.py".to_string(),
        }
    }

    /// Create a TypeScript/Node-style module architecture
    pub fn typescript() -> Self {
        ModuleArchitecture::NodeStyle {
            extensions: vec![".ts".to_string(), ".tsx".to_string(), ".js".to_string()],
            index_name: "index".to_string(),
        }
    }

    /// Create a Go-style module architecture
    pub fn go(gopath: PathBuf) -> Self {
        ModuleArchitecture::GoStyle {
            module_cache: gopath.join("src"),
            extension: "go".to_string(),
        }
    }

    /// Create a Deno/URL-style module architecture
    pub fn deno(cache_dir: PathBuf) -> Self {
        ModuleArchitecture::UrlStyle {
            cache_dir,
            allowed_schemes: vec!["https".to_string(), "http".to_string(), "file".to_string()],
            import_map: HashMap::new(),
        }
    }

    /// Create a Deno-style architecture with an import map
    pub fn deno_with_import_map(cache_dir: PathBuf, import_map: HashMap<String, String>) -> Self {
        ModuleArchitecture::UrlStyle {
            cache_dir,
            allowed_schemes: vec!["https".to_string(), "http".to_string(), "file".to_string()],
            import_map,
        }
    }

    /// Check if this architecture supports URL imports
    pub fn supports_url_imports(&self) -> bool {
        matches!(self, ModuleArchitecture::UrlStyle { .. })
    }

    /// Check if this architecture uses domain-based paths (like Go)
    pub fn uses_domain_paths(&self) -> bool {
        matches!(self, ModuleArchitecture::GoStyle { .. })
    }

    /// Convert a module path to potential filesystem paths
    pub fn module_to_paths(&self, module_path: &[String], base_path: &PathBuf) -> Vec<PathBuf> {
        match self {
            ModuleArchitecture::DotSeparatedPackages { extension } => {
                // com.example.MyClass -> base/com/example/MyClass.hx
                let mut path = base_path.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                vec![path.with_extension(extension)]
            }

            ModuleArchitecture::RustStyle {
                extension,
                mod_file_name,
            } => {
                // module::submodule -> base/module/submodule.rs OR base/module/submodule/mod.rs
                let mut path = base_path.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                vec![path.with_extension(extension), path.join(mod_file_name)]
            }

            ModuleArchitecture::PythonStyle {
                extension,
                init_file_name,
            } => {
                // package.module -> base/package/module.py OR base/package/module/__init__.py
                let mut path = base_path.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                vec![path.with_extension(extension), path.join(init_file_name)]
            }

            ModuleArchitecture::NodeStyle {
                extensions,
                index_name,
            } => {
                // module -> base/module.ts, base/module/index.ts, etc.
                let mut path = base_path.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                let mut paths = Vec::new();
                // Try direct file with each extension
                for ext in extensions {
                    let file_name = format!(
                        "{}{}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        ext
                    );
                    paths.push(path.parent().unwrap_or(&path).join(file_name));
                }
                // Try index file with each extension
                for ext in extensions {
                    paths.push(path.join(format!("{}{}", index_name, ext)));
                }
                paths
            }

            ModuleArchitecture::GoStyle {
                module_cache,
                extension,
            } => {
                // github.com/user/repo/package -> cache/github.com/user/repo/package/*.go
                // Go resolves all .go files in the directory, but we return the directory path
                let mut path = module_cache.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                // In Go, you import packages (directories), not individual files
                // Return the directory and a hypothetical main file
                vec![
                    path.clone(),
                    path.join(format!(
                        "{}.{}",
                        module_path.last().unwrap_or(&"main".to_string()),
                        extension
                    )),
                ]
            }

            ModuleArchitecture::UrlStyle { cache_dir, .. } => {
                // URLs are handled differently - we convert URL to cache path
                // https://deno.land/std@0.200.0/http/server.ts -> cache/deno.land/std@0.200.0/http/server.ts
                // For module_to_paths, we assume the module_path is already the URL path segments
                let mut path = cache_dir.clone();
                for segment in module_path {
                    path = path.join(segment);
                }
                vec![path]
            }

            ModuleArchitecture::Custom { .. } => {
                // Custom resolvers should override resolve_import
                vec![]
            }
        }
    }
}

/// Context provided to import resolvers
#[derive(Debug, Clone)]
pub struct ImportContext {
    /// The current module path (where the import is from)
    pub current_module: Vec<String>,
    /// Search paths for modules (e.g., library paths)
    pub search_paths: Vec<PathBuf>,
    /// Already imported modules (to detect cycles)
    pub imported_modules: Vec<Vec<String>>,
    /// Module aliases (e.g., from package.json or haxelib)
    pub module_aliases: HashMap<String, Vec<String>>,
    /// Whether to allow runtime/extern imports
    pub allow_extern: bool,
    /// Module file architecture
    pub architecture: ModuleArchitecture,
    /// Source root directories (for relative imports)
    pub source_roots: Vec<PathBuf>,
}

impl ImportContext {
    pub fn new() -> Self {
        Self {
            current_module: Vec::new(),
            search_paths: Vec::new(),
            imported_modules: Vec::new(),
            module_aliases: HashMap::new(),
            allow_extern: true,
            architecture: ModuleArchitecture::default(),
            source_roots: Vec::new(),
        }
    }

    pub fn with_current_module(mut self, path: Vec<String>) -> Self {
        self.current_module = path;
        self
    }

    pub fn with_search_path(mut self, path: PathBuf) -> Self {
        self.search_paths.push(path);
        self
    }

    pub fn with_alias(mut self, alias: String, target: Vec<String>) -> Self {
        self.module_aliases.insert(alias, target);
        self
    }

    pub fn with_architecture(mut self, arch: ModuleArchitecture) -> Self {
        self.architecture = arch;
        self
    }

    pub fn with_source_root(mut self, root: PathBuf) -> Self {
        self.source_roots.push(root);
        self
    }

    /// Find all potential file paths for a module path
    pub fn find_module_paths(&self, module_path: &[String]) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // Check in source roots first
        for root in &self.source_roots {
            paths.extend(self.architecture.module_to_paths(module_path, root));
        }

        // Then check in search paths (library paths)
        for search in &self.search_paths {
            paths.extend(self.architecture.module_to_paths(module_path, search));
        }

        paths
    }
}

impl Default for ImportContext {
    fn default() -> Self {
        Self::new()
    }
}

/// The main trait for implementing import resolution
///
/// Different language frontends implement this trait to provide
/// their own import resolution logic.
pub trait ImportResolver: Send + Sync {
    /// Resolve an import declaration
    ///
    /// Returns the resolved import or an error
    fn resolve_import(
        &self,
        import: &TypedImport,
        context: &ImportContext,
    ) -> Result<Vec<ResolvedImport>, ImportError>;

    /// Resolve a module path to see if it exists
    ///
    /// Used for validating import paths before full resolution
    fn module_exists(&self, path: &[String], context: &ImportContext) -> bool;

    /// Get all exports from a module
    ///
    /// Used for IDE autocomplete and glob imports
    fn get_module_exports(
        &self,
        path: &[String],
        context: &ImportContext,
    ) -> Result<Vec<ExportedSymbol>, ImportError>;

    /// Resolve a single qualified name to a type
    ///
    /// Used when encountering a fully qualified type reference
    fn resolve_qualified_type(
        &self,
        path: &[String],
        context: &ImportContext,
    ) -> Result<Type, ImportError>;

    /// Get the name of this resolver (for debugging/logging)
    fn resolver_name(&self) -> &str;

    /// Check if this resolver handles a given module path
    ///
    /// Used when multiple resolvers are registered to determine
    /// which one should handle a particular import
    fn handles_module(&self, path: &[String]) -> bool {
        // Default: handle all modules
        true
    }
}

/// A resolver that combines multiple resolvers
pub struct ChainedResolver {
    resolvers: Vec<Box<dyn ImportResolver>>,
}

impl ChainedResolver {
    pub fn new() -> Self {
        Self {
            resolvers: Vec::new(),
        }
    }

    pub fn add_resolver(mut self, resolver: Box<dyn ImportResolver>) -> Self {
        self.resolvers.push(resolver);
        self
    }
}

impl Default for ChainedResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportResolver for ChainedResolver {
    fn resolve_import(
        &self,
        import: &TypedImport,
        context: &ImportContext,
    ) -> Result<Vec<ResolvedImport>, ImportError> {
        let path: Vec<String> = import
            .module_path
            .iter()
            .map(|s| s.resolve_global().unwrap_or_default())
            .collect();

        for resolver in &self.resolvers {
            if resolver.handles_module(&path) {
                return resolver.resolve_import(import, context);
            }
        }

        Err(ImportError::ModuleNotFound {
            path,
            span: import.span,
            suggestions: Vec::new(),
        })
    }

    fn module_exists(&self, path: &[String], context: &ImportContext) -> bool {
        self.resolvers
            .iter()
            .any(|r| r.module_exists(path, context))
    }

    fn get_module_exports(
        &self,
        path: &[String],
        context: &ImportContext,
    ) -> Result<Vec<ExportedSymbol>, ImportError> {
        for resolver in &self.resolvers {
            if resolver.handles_module(path) {
                return resolver.get_module_exports(path, context);
            }
        }

        Err(ImportError::ModuleNotFound {
            path: path.to_vec(),
            span: Span::new(0, 0),
            suggestions: Vec::new(),
        })
    }

    fn resolve_qualified_type(
        &self,
        path: &[String],
        context: &ImportContext,
    ) -> Result<Type, ImportError> {
        for resolver in &self.resolvers {
            if resolver.handles_module(path) {
                return resolver.resolve_qualified_type(path, context);
            }
        }

        Err(ImportError::ModuleNotFound {
            path: path.to_vec(),
            span: Span::new(0, 0),
            suggestions: Vec::new(),
        })
    }

    fn resolver_name(&self) -> &str {
        "ChainedResolver"
    }
}

/// A simple in-memory resolver for testing and builtin types
pub struct BuiltinResolver {
    /// Registered builtin types
    types: HashMap<Vec<String>, Type>,
    /// Registered builtin modules
    modules: HashMap<Vec<String>, Vec<ExportedSymbol>>,
}

impl BuiltinResolver {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    pub fn register_type(mut self, path: Vec<String>, ty: Type) -> Self {
        self.types.insert(path, ty);
        self
    }

    pub fn register_module(mut self, path: Vec<String>, exports: Vec<ExportedSymbol>) -> Self {
        self.modules.insert(path, exports);
        self
    }
}

impl Default for BuiltinResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportResolver for BuiltinResolver {
    fn resolve_import(
        &self,
        import: &TypedImport,
        _context: &ImportContext,
    ) -> Result<Vec<ResolvedImport>, ImportError> {
        let path: Vec<String> = import
            .module_path
            .iter()
            .map(|s| s.resolve_global().unwrap_or_default())
            .collect();

        if let Some(ty) = self.types.get(&path) {
            return Ok(vec![ResolvedImport::Type {
                qualified_name: path,
                ty: ty.clone(),
                is_extern: true,
            }]);
        }

        if let Some(exports) = self.modules.get(&path) {
            return Ok(vec![ResolvedImport::Module {
                path: path.clone(),
                exports: exports.clone(),
            }]);
        }

        Err(ImportError::ModuleNotFound {
            path,
            span: import.span,
            suggestions: Vec::new(),
        })
    }

    fn module_exists(&self, path: &[String], _context: &ImportContext) -> bool {
        self.modules.contains_key(path) || self.types.contains_key(path)
    }

    fn get_module_exports(
        &self,
        path: &[String],
        _context: &ImportContext,
    ) -> Result<Vec<ExportedSymbol>, ImportError> {
        self.modules
            .get(path)
            .cloned()
            .ok_or_else(|| ImportError::ModuleNotFound {
                path: path.to_vec(),
                span: Span::new(0, 0),
                suggestions: Vec::new(),
            })
    }

    fn resolve_qualified_type(
        &self,
        path: &[String],
        _context: &ImportContext,
    ) -> Result<Type, ImportError> {
        self.types
            .get(path)
            .cloned()
            .ok_or_else(|| ImportError::ModuleNotFound {
                path: path.to_vec(),
                span: Span::new(0, 0),
                suggestions: Vec::new(),
            })
    }

    fn resolver_name(&self) -> &str {
        "BuiltinResolver"
    }
}

/// Import resolution manager that tracks resolved imports
pub struct ImportManager {
    /// The resolver to use
    resolver: Box<dyn ImportResolver>,
    /// Cache of resolved imports
    cache: HashMap<Vec<String>, Vec<ResolvedImport>>,
    /// Symbol table built from imports
    symbol_table: HashMap<String, ResolvedImport>,
}

impl ImportManager {
    pub fn new(resolver: Box<dyn ImportResolver>) -> Self {
        Self {
            resolver,
            cache: HashMap::new(),
            symbol_table: HashMap::new(),
        }
    }

    /// Process an import and add it to the symbol table
    pub fn process_import(
        &mut self,
        import: &TypedImport,
        context: &ImportContext,
    ) -> Result<(), ImportError> {
        let path: Vec<String> = import
            .module_path
            .iter()
            .map(|s| s.resolve_global().unwrap_or_default())
            .collect();

        // Check cache first
        if let Some(cached) = self.cache.get(&path).cloned() {
            for resolved in cached {
                self.add_to_symbol_table(resolved, import)?;
            }
            return Ok(());
        }

        // Resolve the import
        let resolved = self.resolver.resolve_import(import, context)?;

        // Add to cache
        self.cache.insert(path, resolved.clone());

        // Add to symbol table
        for r in resolved {
            self.add_to_symbol_table(r, import)?;
        }

        Ok(())
    }

    fn add_to_symbol_table(
        &mut self,
        resolved: ResolvedImport,
        import: &TypedImport,
    ) -> Result<(), ImportError> {
        use crate::typed_ast::TypedImportItem;

        for item in &import.items {
            match item {
                TypedImportItem::Named { name, alias } => {
                    let symbol_name = alias
                        .as_ref()
                        .unwrap_or(name)
                        .resolve_global()
                        .unwrap_or_default();
                    self.symbol_table.insert(symbol_name, resolved.clone());
                }
                TypedImportItem::Glob => {
                    // For glob imports, add all exports
                    if let ResolvedImport::Module { exports, .. } = &resolved {
                        for export in exports {
                            let export_resolved = match &export.kind {
                                SymbolKind::Type
                                | SymbolKind::Class
                                | SymbolKind::Enum
                                | SymbolKind::Interface
                                | SymbolKind::Trait => {
                                    ResolvedImport::Type {
                                        qualified_name: import
                                            .module_path
                                            .iter()
                                            .map(|s| s.resolve_global().unwrap_or_default())
                                            .chain(std::iter::once(export.name.clone()))
                                            .collect(),
                                        ty: Type::Never, // Placeholder, would need actual resolution
                                        is_extern: false,
                                    }
                                }
                                SymbolKind::Function => ResolvedImport::Function {
                                    qualified_name: import
                                        .module_path
                                        .iter()
                                        .map(|s| s.resolve_global().unwrap_or_default())
                                        .chain(std::iter::once(export.name.clone()))
                                        .collect(),
                                    params: Vec::new(),
                                    return_type: Type::Never,
                                    is_extern: false,
                                },
                                SymbolKind::Constant => ResolvedImport::Constant {
                                    qualified_name: import
                                        .module_path
                                        .iter()
                                        .map(|s| s.resolve_global().unwrap_or_default())
                                        .chain(std::iter::once(export.name.clone()))
                                        .collect(),
                                    ty: Type::Never,
                                },
                                SymbolKind::Module => continue,
                            };
                            self.symbol_table
                                .insert(export.name.clone(), export_resolved);
                        }
                    }
                }
                TypedImportItem::Default(name) => {
                    self.symbol_table
                        .insert(name.resolve_global().unwrap_or_default(), resolved.clone());
                }
            }
        }

        Ok(())
    }

    /// Look up a symbol by name
    pub fn lookup(&self, name: &str) -> Option<&ResolvedImport> {
        self.symbol_table.get(name)
    }

    /// Check if a symbol is imported
    pub fn is_imported(&self, name: &str) -> bool {
        self.symbol_table.contains_key(name)
    }

    /// Get all imported symbols
    pub fn all_symbols(&self) -> impl Iterator<Item = (&String, &ResolvedImport)> {
        self.symbol_table.iter()
    }
}

/// Entry point resolution based on module architecture
///
/// Different languages have different conventions for specifying entry points:
/// - Haxe: `ClassName.staticMethod` (e.g., `Test.main`)
/// - Java: `package.ClassName.main` (e.g., `com.example.Main.main`)
/// - Rust: `module::function` (e.g., `main` or `app::main`)
/// - Python: `module.function` (e.g., `main.run`)
/// - Go: `package.Function` (e.g., `main.main`)
#[derive(Debug, Clone)]
pub struct EntryPointResolver {
    /// The module architecture
    architecture: ModuleArchitecture,
}

impl EntryPointResolver {
    /// Create a new entry point resolver with the given architecture
    pub fn new(architecture: ModuleArchitecture) -> Self {
        Self { architecture }
    }

    /// Create a Haxe-style resolver
    pub fn haxe() -> Self {
        Self::new(ModuleArchitecture::haxe())
    }

    /// Create a Java-style resolver
    pub fn java() -> Self {
        Self::new(ModuleArchitecture::java())
    }

    /// Create a Rust-style resolver
    pub fn rust() -> Self {
        Self::new(ModuleArchitecture::rust())
    }

    /// Resolve an entry point string to potential function name candidates
    ///
    /// Takes an entry point string (e.g., "Test.main") and returns a list of
    /// potential function names to search for in the compiled module.
    ///
    /// The candidates are ordered by preference (most likely first).
    pub fn resolve_candidates(&self, entry_point: &str) -> Vec<String> {
        match &self.architecture {
            ModuleArchitecture::DotSeparatedPackages { .. } => {
                // Haxe/Java style: Test.main -> ["Test_main", "Test.main", "main"]
                // The underscore variant is commonly used in compiled output
                self.resolve_haxe_style(entry_point)
            }
            ModuleArchitecture::RustStyle { .. } => {
                // Rust style: module::main -> ["module::main", "main"]
                self.resolve_rust_style(entry_point)
            }
            ModuleArchitecture::PythonStyle { .. } => {
                // Python style: module.main -> ["module.main", "module_main", "main"]
                self.resolve_python_style(entry_point)
            }
            ModuleArchitecture::NodeStyle { .. } => {
                // Node/TS style: similar to Python
                self.resolve_node_style(entry_point)
            }
            ModuleArchitecture::GoStyle { .. } => {
                // Go style: package.Function -> ["package.Function", "Function"]
                self.resolve_go_style(entry_point)
            }
            ModuleArchitecture::UrlStyle { .. } => {
                // Deno style: URL imports, entry is just a function name
                vec![entry_point.to_string(), "main".to_string()]
            }
            ModuleArchitecture::Custom { .. } => {
                // Custom: just use the entry point as-is
                vec![entry_point.to_string()]
            }
        }
    }

    /// Parse entry point into parts based on the architecture's separator
    pub fn parse_entry_point<'a>(&self, entry_point: &'a str) -> Vec<&'a str> {
        match &self.architecture {
            ModuleArchitecture::RustStyle { .. } => {
                // Rust uses :: as separator
                entry_point.split("::").collect()
            }
            _ => {
                // Most languages use . as separator
                entry_point.split('.').collect()
            }
        }
    }

    /// Resolve Haxe-style entry point (ClassName.methodName)
    fn resolve_haxe_style(&self, entry_point: &str) -> Vec<String> {
        let parts: Vec<&str> = entry_point.split('.').collect();
        let mut candidates = Vec::new();

        if parts.len() >= 2 {
            // Primary: ClassName_methodName (most common in compiled output)
            candidates.push(parts.join("_"));
            // Also try: ClassName.methodName (as-is)
            candidates.push(entry_point.to_string());
            // Also try: just the method name
            if let Some(method) = parts.last() {
                candidates.push(method.to_string());
            }
        } else {
            // Single name - use as-is
            candidates.push(entry_point.to_string());
        }

        // Always include "main" as ultimate fallback
        if !candidates.contains(&"main".to_string()) {
            candidates.push("main".to_string());
        }

        candidates
    }

    /// Resolve Rust-style entry point (module::function)
    fn resolve_rust_style(&self, entry_point: &str) -> Vec<String> {
        let parts: Vec<&str> = entry_point.split("::").collect();
        let mut candidates = Vec::new();

        // Primary: exact match
        candidates.push(entry_point.to_string());

        if parts.len() >= 2 {
            // Also try just the function name
            if let Some(func) = parts.last() {
                candidates.push(func.to_string());
            }
        }

        // Always include "main" as fallback
        if !candidates.contains(&"main".to_string()) {
            candidates.push("main".to_string());
        }

        candidates
    }

    /// Resolve Python-style entry point (module.function)
    fn resolve_python_style(&self, entry_point: &str) -> Vec<String> {
        let parts: Vec<&str> = entry_point.split('.').collect();
        let mut candidates = Vec::new();

        // Primary: exact match
        candidates.push(entry_point.to_string());

        if parts.len() >= 2 {
            // Also try underscore variant
            candidates.push(parts.join("_"));
            // Also try just the function name
            if let Some(func) = parts.last() {
                candidates.push(func.to_string());
            }
        }

        // Always include "main" as fallback
        if !candidates.contains(&"main".to_string()) {
            candidates.push("main".to_string());
        }

        candidates
    }

    /// Resolve Node/TypeScript-style entry point
    fn resolve_node_style(&self, entry_point: &str) -> Vec<String> {
        // Similar to Python style
        self.resolve_python_style(entry_point)
    }

    /// Resolve Go-style entry point (package.Function)
    fn resolve_go_style(&self, entry_point: &str) -> Vec<String> {
        let parts: Vec<&str> = entry_point.split('.').collect();
        let mut candidates = Vec::new();

        // Primary: exact match
        candidates.push(entry_point.to_string());

        if parts.len() >= 2 {
            // Go uses PascalCase for exported functions
            if let Some(func) = parts.last() {
                candidates.push(func.to_string());
            }
        }

        // Always include "main" as fallback (Go programs must have main)
        if !candidates.contains(&"main".to_string()) {
            candidates.push("main".to_string());
        }

        candidates
    }

    /// Get the default entry point for this architecture
    pub fn default_entry(&self) -> &str {
        match &self.architecture {
            ModuleArchitecture::GoStyle { .. } => "main.main",
            _ => "main",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::AstArena;
    use crate::typed_ast::TypedImportItem;

    #[test]
    fn test_builtin_resolver() {
        let resolver = BuiltinResolver::new().register_type(
            vec!["std".to_string(), "math".to_string(), "sqrt".to_string()],
            Type::Primitive(crate::PrimitiveType::F64),
        );

        let context = ImportContext::new();
        assert!(resolver.module_exists(
            &["std".to_string(), "math".to_string(), "sqrt".to_string()],
            &context
        ));
    }

    #[test]
    fn test_import_manager() {
        let resolver = BuiltinResolver::new().register_type(
            vec![
                "haxe".to_string(),
                "ds".to_string(),
                "StringMap".to_string(),
            ],
            Type::Primitive(crate::PrimitiveType::I64), // Placeholder type
        );

        let mut manager = ImportManager::new(Box::new(resolver));
        let context = ImportContext::new();

        let mut arena = AstArena::new();
        let import = TypedImport {
            module_path: vec![
                arena.intern_string("haxe"),
                arena.intern_string("ds"),
                arena.intern_string("StringMap"),
            ],
            items: vec![TypedImportItem::Named {
                name: arena.intern_string("StringMap"),
                alias: None,
            }],
            span: Span::new(0, 10),
        };

        let result = manager.process_import(&import, &context);
        assert!(result.is_ok());
        assert!(manager.is_imported("StringMap"));
    }
}
