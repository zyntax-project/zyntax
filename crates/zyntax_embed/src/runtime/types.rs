//! What a runtime hands back, and what it is handed.
//!
//! The error and event types, the native signature a compiled function
//! is called through, and the callbacks a host resolves imports with.

use crate::error::{ConversionError, ZyntaxError};
use crate::grammar::GrammarError;
use zyntax_compiler::CompilerError;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Errors that can occur during runtime operations
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Compilation error: {0}")]
    Compilation(#[from] CompilerError),

    #[error("Function not found: {0}")]
    FunctionNotFound(String),

    #[error("Type conversion error: {0}")]
    Conversion(#[from] ConversionError),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Promise error: {0}")]
    Promise(String),

    #[error("Invalid argument count: expected {expected}, got {got}")]
    ArgumentCount { expected: usize, got: usize },
}

impl From<ZyntaxError> for RuntimeError {
    fn from(err: ZyntaxError) -> Self {
        RuntimeError::Execution(err.to_string())
    }
}

/// Runtime-side semantic events emitted from language constructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvent {
    Render {
        value: String,
        options: Vec<(String, String)>,
    },
    Stream {
        pipeline: String,
        stage_count: usize,
    },
    /// A hot reload was applied. One event per reload, carrying the
    /// per-function outcomes — the observable boundary a framework
    /// subscribes to for invalidation.
    Reload {
        reloaded: Vec<String>,
        added: Vec<String>,
        dispatch_patched: Vec<String>,
        failed: Vec<(String, String)>,
    },
}

// ============================================================================
// Native Calling Convention Types
// ============================================================================

/// Native type for function signatures
///
/// Represents the primitive types that can be passed to/from JIT-compiled functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeType {
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Boolean (passed as i8)
    Bool,
    /// Void (no return value)
    Void,
    /// Pointer (passed as usize)
    Ptr,
}

impl NativeType {
    /// Convert from HIR type to native type
    pub fn from_hir_type(ty: &zyntax_compiler::hir::HirType) -> Self {
        use zyntax_compiler::hir::HirType;
        match ty {
            HirType::I8 | HirType::I16 | HirType::I32 => NativeType::I32,
            HirType::I64 | HirType::I128 => NativeType::I64,
            HirType::U8 | HirType::U16 | HirType::U32 => NativeType::I32,
            HirType::U64 | HirType::U128 => NativeType::I64,
            HirType::F32 => NativeType::F32,
            HirType::F64 => NativeType::F64,
            HirType::Bool => NativeType::Bool,
            HirType::Void => NativeType::Void,
            HirType::Ptr(_) | HirType::Ref { .. } | HirType::Function(_) => NativeType::Ptr,
            HirType::Promise(_) => NativeType::Ptr,
            _ => NativeType::I64, // Default to i64 for unknown types
        }
    }
}

/// Function signature for native calling convention
///
/// Describes the parameter types and return type for a JIT-compiled function.
#[derive(Debug, Clone)]
pub struct NativeSignature {
    /// Parameter types
    pub params: Vec<NativeType>,
    /// Return type
    pub ret: NativeType,
}

impl NativeSignature {
    /// Create a new signature
    pub fn new(params: &[NativeType], ret: NativeType) -> Self {
        Self {
            params: params.to_vec(),
            ret,
        }
    }

    /// Create a signature from an HIR function signature
    pub fn from_hir_signature(sig: &zyntax_compiler::hir::HirFunctionSignature) -> Self {
        let params: Vec<NativeType> = sig
            .params
            .iter()
            .map(|p| NativeType::from_hir_type(&p.ty))
            .collect();

        let ret = sig
            .returns
            .first()
            .map(|ty| NativeType::from_hir_type(ty))
            .unwrap_or(NativeType::Void);

        Self { params, ret }
    }

    /// Create a signature from a string like "(i32, i32) -> i32"
    pub fn parse(s: &str) -> Option<Self> {
        // Simple parser for signature strings
        let s = s.trim();

        // Find the arrow
        let arrow_pos = s.find("->")?;
        let params_str = s[..arrow_pos].trim();
        let ret_str = s[arrow_pos + 2..].trim();

        // Parse return type
        let ret = Self::parse_type(ret_str)?;

        // Parse parameters
        let params_str = params_str.strip_prefix('(')?.strip_suffix(')')?;
        let params: Option<Vec<_>> = if params_str.is_empty() {
            Some(vec![])
        } else {
            params_str
                .split(',')
                .map(|p| Self::parse_type(p.trim()))
                .collect()
        };

        Some(Self {
            params: params?,
            ret,
        })
    }

    fn parse_type(s: &str) -> Option<NativeType> {
        match s {
            "i32" => Some(NativeType::I32),
            "i64" => Some(NativeType::I64),
            "f32" => Some(NativeType::F32),
            "f64" => Some(NativeType::F64),
            "bool" => Some(NativeType::Bool),
            "void" | "()" => Some(NativeType::Void),
            "ptr" | "*" => Some(NativeType::Ptr),
            _ => None,
        }
    }
}

/// Simple callback type for resolving imports
///
/// Called during compilation when an import statement is encountered.
/// Returns the resolved module content (source code) or an error message.
///
/// # Arguments
/// * `module_path` - The import path as a dot-separated string (e.g., "std.io", "my_module")
///
/// # Returns
/// * `Ok(Some(source))` - The resolved module source code
/// * `Ok(None)` - Module not found by this resolver (try next resolver)
/// * `Err(message)` - Error resolving the module
pub type ImportResolverCallback = Box<dyn Fn(&str) -> Result<Option<String>, String> + Send + Sync>;

/// Callback for imports parsed into build-time artifacts.
///
/// Compiled resolvers run before source resolvers so production deployments
/// can keep source fallbacks for development without paying their parse cost.
pub type CompiledImportResolverCallback =
    Box<dyn Fn(&str) -> Result<Option<crate::CompiledImport>, String> + Send + Sync>;

// Re-export the full ImportResolver trait from the compiler for advanced use cases

pub use zyntax_compiler::{
    BuiltinResolver, ChainedResolver, ExportedSymbol, ImportContext, ImportError, ImportManager,
    ImportResolver as ImportResolverTrait, ModuleArchitecture, ResolvedImport, SymbolKind,
};
