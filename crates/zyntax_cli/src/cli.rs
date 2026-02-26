//! CLI argument parsing and command definitions

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "zyntax")]
#[command(about = "Zyntax compiler - Compile typed AST to native code", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

/// Module resolver architecture type for import resolution
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq)]
pub enum ModuleArch {
    /// Java/Haxe style: com.example.MyClass -> com/example/MyClass.hx
    #[default]
    Haxe,
    /// Java style packages
    Java,
    /// Rust style with mod.rs
    Rust,
    /// Python style with __init__.py
    Python,
    /// TypeScript/Node style with index files
    Typescript,
    /// Go style with domain-based imports
    Go,
    /// Deno style with URL imports
    Deno,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Compile source files to native code
    Compile {
        /// Input file(s) or directory (.json for TypedAST, .zbc for HIR bytecode)
        /// For grammar-based compilation, use --source and --grammar instead
        #[arg(value_name = "INPUT", num_args = 0..)]
        input: Vec<PathBuf>,

        /// Source code file to compile (used with --grammar for ZynPEG-based parsing)
        #[arg(short, long, value_name = "SOURCE")]
        source: Option<PathBuf>,

        /// ZynPEG grammar file (.zyn) defining the language parser
        #[arg(short, long, value_name = "GRAMMAR")]
        grammar: Option<PathBuf>,

        /// Output file path
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Backend to use (cranelift, llvm)
        #[arg(short, long, default_value = "cranelift")]
        backend: String,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,

        /// Input format (auto, typed-ast, hir-bytecode, zyn)
        /// - auto: Auto-detect from file extension
        /// - typed-ast: TypedAST JSON files
        /// - hir-bytecode: HIR bytecode (.zbc) files
        /// - zyn: ZynPEG grammar-based parsing (requires --grammar and --source)
        #[arg(short = 'f', long, default_value = "auto")]
        format: String,

        /// JIT compile and run the program immediately
        #[arg(long)]
        jit: bool,

        /// Entry point for JIT execution (e.g., "main", "MyClass.main", "com.example.Main.run")
        /// Uses the module resolver style for path resolution.
        /// Defaults to "main" or the grammar's @language.entry_point if defined.
        #[arg(long = "entry", value_name = "ENTRY")]
        entry_point: Option<String>,

        // === Import Resolution Options ===
        /// Module resolver architecture for import resolution
        #[arg(long = "resolver", value_enum, default_value = "haxe")]
        resolver: ModuleArch,

        /// Source root directory for resolving imports (can be specified multiple times)
        #[arg(long = "source-root", value_name = "DIR")]
        source_roots: Vec<PathBuf>,

        /// Library search path for imports (can be specified multiple times)
        #[arg(short = 'L', long = "lib-path", value_name = "DIR")]
        lib_paths: Vec<PathBuf>,

        /// Import map file (JSON) for URL-style imports (Deno)
        #[arg(long = "import-map", value_name = "FILE")]
        import_map: Option<PathBuf>,

        /// Cache directory for compiled modules (incremental compilation)
        #[arg(long = "cache-dir", value_name = "DIR")]
        cache_dir: Option<PathBuf>,

        /// Disable incremental compilation cache
        #[arg(long)]
        no_cache: bool,

        /// Use legacy Grammar1 runtime (ZpegCompiler + pest_vm)
        /// Default is Grammar2 (GrammarInterpreter with named bindings)
        #[arg(long)]
        grammar1: bool,

        // === Runtime Library Options ===
        /// ZPack archive(s) to load runtime symbols from (.zpack files)
        /// For JIT mode: provides runtime symbols dynamically
        /// Can be specified multiple times for multiple runtime libraries
        #[arg(short = 'P', long = "pack", value_name = "ZPACK")]
        packs: Vec<PathBuf>,

        /// Static library to link for AOT compilation
        /// Accepts: full path (/path/to/lib.a), library name (foo), or file name (libfoo.a)
        /// Library names are searched in standard system paths (/usr/local/lib, /usr/lib, etc.)
        /// Can be specified multiple times for multiple libraries
        #[arg(long = "lib", value_name = "LIB")]
        static_libs: Vec<PathBuf>,
    },

    /// Start an interactive REPL with a ZynPEG grammar
    Repl {
        /// ZynPEG grammar file (.zyn) defining the language parser
        #[arg(short, long, value_name = "GRAMMAR")]
        grammar: PathBuf,

        /// Backend to use (cranelift, llvm)
        #[arg(short, long, default_value = "cranelift")]
        backend: String,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "0")]
        opt_level: u8,

        /// Module resolver architecture for import resolution
        #[arg(long = "resolver", value_enum, default_value = "haxe")]
        resolver: ModuleArch,

        /// Source root directory for resolving imports
        #[arg(long = "source-root", value_name = "DIR")]
        source_roots: Vec<PathBuf>,

        /// Library search path for imports
        #[arg(short = 'L', long = "lib-path", value_name = "DIR")]
        lib_paths: Vec<PathBuf>,
    },

    /// Manage the compilation cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Create or inspect ZPack archives (modules + runtime)
    Pack {
        #[command(subcommand)]
        action: PackAction,
    },

    /// Display version information
    Version,
}

#[derive(Subcommand)]
pub enum CacheAction {
    /// Clear the compilation cache
    Clear {
        /// Cache directory (defaults to ~/.zyntax/cache)
        #[arg(long = "cache-dir", value_name = "DIR")]
        cache_dir: Option<PathBuf>,
    },

    /// Show cache statistics
    Stats {
        /// Cache directory (defaults to ~/.zyntax/cache)
        #[arg(long = "cache-dir", value_name = "DIR")]
        cache_dir: Option<PathBuf>,
    },

    /// List cached modules
    List {
        /// Cache directory (defaults to ~/.zyntax/cache)
        #[arg(long = "cache-dir", value_name = "DIR")]
        cache_dir: Option<PathBuf>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Subcommand)]
pub enum PackAction {
    /// Create a new ZPack archive from modules and runtime libraries
    Create {
        /// Output .zpack file path
        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,

        /// Package name
        #[arg(short, long)]
        name: String,

        /// Package version (semver)
        #[arg(long, default_value = "0.1.0")]
        version: String,

        /// Source language (e.g., "haxe", "zig")
        #[arg(long, default_value = "haxe")]
        language: String,

        /// HIR bytecode modules to include (.zbc files or directories)
        #[arg(short, long = "module", value_name = "PATH")]
        modules: Vec<PathBuf>,

        /// Runtime library for a target (format: TARGET:PATH, e.g., x86_64-apple-darwin:/path/to/runtime.zrtl)
        #[arg(short, long = "runtime", value_name = "TARGET:PATH")]
        runtimes: Vec<String>,

        /// Package description
        #[arg(long)]
        description: Option<String>,

        /// Entry point module path
        #[arg(long = "entry")]
        entry_point: Option<String>,
    },

    /// List contents of a ZPack archive
    List {
        /// ZPack file to inspect
        #[arg(value_name = "ZPACK")]
        zpack: PathBuf,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Extract a ZPack archive
    Extract {
        /// ZPack file to extract
        #[arg(value_name = "ZPACK")]
        zpack: PathBuf,

        /// Output directory (defaults to current directory)
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
    },

    /// Show information about the current platform target
    Target,
}

impl Commands {
    pub fn compile_args(&self) -> Option<CompileArgs> {
        match self {
            Commands::Compile {
                input,
                source,
                grammar,
                output,
                backend,
                opt_level,
                format,
                jit,
                entry_point,
                resolver,
                source_roots,
                lib_paths,
                import_map,
                cache_dir,
                no_cache,
                grammar1,
                packs,
                static_libs,
            } => Some(CompileArgs {
                input: input.clone(),
                source: source.clone(),
                grammar: grammar.clone(),
                output: output.clone(),
                backend: backend.clone(),
                opt_level: *opt_level,
                format: format.clone(),
                jit: *jit,
                entry_point: entry_point.clone(),
                resolver: *resolver,
                source_roots: source_roots.clone(),
                lib_paths: lib_paths.clone(),
                import_map: import_map.clone(),
                cache_dir: cache_dir.clone(),
                no_cache: *no_cache,
                grammar1: *grammar1,
                packs: packs.clone(),
                static_libs: static_libs.clone(),
            }),
            _ => None,
        }
    }

    pub fn repl_args(&self) -> Option<ReplArgs> {
        match self {
            Commands::Repl {
                grammar,
                backend,
                opt_level,
                resolver,
                source_roots,
                lib_paths,
            } => Some(ReplArgs {
                grammar: grammar.clone(),
                backend: backend.clone(),
                opt_level: *opt_level,
                resolver: *resolver,
                source_roots: source_roots.clone(),
                lib_paths: lib_paths.clone(),
            }),
            _ => None,
        }
    }

    pub fn cache_args(&self) -> Option<&CacheAction> {
        match self {
            Commands::Cache { action } => Some(action),
            _ => None,
        }
    }

    pub fn pack_args(&self) -> Option<&PackAction> {
        match self {
            Commands::Pack { action } => Some(action),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileArgs {
    pub input: Vec<PathBuf>,
    pub source: Option<PathBuf>,
    pub grammar: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub backend: String,
    pub opt_level: u8,
    pub format: String,
    pub jit: bool,
    pub entry_point: Option<String>,
    pub resolver: ModuleArch,
    pub source_roots: Vec<PathBuf>,
    pub lib_paths: Vec<PathBuf>,
    pub import_map: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub no_cache: bool,
    pub grammar1: bool,
    pub packs: Vec<PathBuf>,
    pub static_libs: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ReplArgs {
    pub grammar: PathBuf,
    pub backend: String,
    pub opt_level: u8,
    pub resolver: ModuleArch,
    pub source_roots: Vec<PathBuf>,
    pub lib_paths: Vec<PathBuf>,
}

/// Get the default cache directory
pub fn default_cache_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".zyntax").join("cache"))
        .unwrap_or_else(|| PathBuf::from(".zyntax_cache"))
}
