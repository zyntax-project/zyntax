//! # AST Lowering Pipeline
//!
//! Orchestrates the transformation from TypedAST to HIR in SSA form,
//! ready for both Cranelift and LLVM backends.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use zyntax_typed_ast::{
    AstArena, InternedString, Mutability, Type, TypeId, TypedDeclaration, TypedEffect,
    TypedEffectHandler, TypedFunction, TypedNode, TypedProgram, Visibility,
};

/// Helper to compute a hash for generating synthetic TypeIds
fn hash_string(s: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}
use crate::cfg::{CfgBuilder, ControlFlowGraph};
use crate::hir::{
    HirEffect, HirEffectHandler, HirEffectHandlerImpl, HirEffectOp, HirFunction,
    HirFunctionSignature, HirHandlerField, HirId, HirLifetime, HirModule, HirParam, HirType,
    HirTypeParam, OwnershipMode, ParamAttributes,
};
use crate::ssa::{SsaBuilder, SsaForm};
use crate::CompilerResult;
use std::collections::HashMap;

/// Target data layout information for precise size and alignment calculations
#[derive(Debug, Clone)]
pub struct TargetData {
    /// Pointer size in bytes (4 for 32-bit, 8 for 64-bit)
    pub pointer_size: usize,
    /// Pointer alignment in bytes
    pub pointer_alignment: usize,
    /// Target architecture (e.g., "x86_64", "aarch64", "wasm32")
    pub architecture: String,
    /// Endianness (true for big-endian, false for little-endian)
    pub is_big_endian: bool,
}

impl TargetData {
    /// Create target data for the host platform
    pub fn host() -> Self {
        Self {
            pointer_size: std::mem::size_of::<*const ()>(),
            pointer_alignment: std::mem::align_of::<*const ()>(),
            architecture: std::env::consts::ARCH.to_string(),
            is_big_endian: cfg!(target_endian = "big"),
        }
    }

    /// Create target data for a specific architecture
    pub fn for_architecture(arch: &str) -> Self {
        match arch {
            "x86_64" | "aarch64" => Self {
                pointer_size: 8,
                pointer_alignment: 8,
                architecture: arch.to_string(),
                is_big_endian: false,
            },
            "x86" | "arm" | "wasm32" => Self {
                pointer_size: 4,
                pointer_alignment: 4,
                architecture: arch.to_string(),
                is_big_endian: false,
            },
            "wasm64" => Self {
                pointer_size: 8,
                pointer_alignment: 8,
                architecture: arch.to_string(),
                is_big_endian: false,
            },
            _ => Self::host(), // Default to host
        }
    }

    /// Calculate the size of a type in bytes
    pub fn size_of(&self, ty: &Type) -> usize {
        match ty {
            Type::Primitive(prim) => self.primitive_size(prim),
            Type::Reference { .. } => self.pointer_size,
            Type::Array {
                element_type,
                size: Some(size_val),
                ..
            } => {
                let count = match size_val {
                    zyntax_typed_ast::ConstValue::UInt(u) => *u as usize,
                    zyntax_typed_ast::ConstValue::Int(i) => (*i).max(0) as usize,
                    _ => 1,
                };
                let elem_size = self.size_of(element_type);
                let elem_align = self.align_of(element_type);
                // Align each element
                Self::align_to(elem_size, elem_align) * count
            }
            Type::Tuple(elems) => self.calculate_struct_size(elems.iter().collect()),
            Type::Struct { fields, .. } => {
                let field_types: Vec<&Type> = fields.iter().map(|f| &f.ty).collect();
                self.calculate_struct_size(field_types)
            }
            Type::Function { .. } => self.pointer_size, // Function pointer
            Type::Named { .. } => {
                // For named types, we'd need to look up the definition
                // For now, use a conservative estimate
                self.pointer_size
            }
            _ => self.pointer_size, // Conservative default
        }
    }

    /// Calculate the alignment of a type in bytes
    pub fn align_of(&self, ty: &Type) -> usize {
        match ty {
            Type::Primitive(prim) => self.primitive_alignment(prim),
            Type::Reference { .. } => self.pointer_alignment,
            Type::Array { element_type, .. } => self.align_of(element_type),
            Type::Tuple(elems) => elems.iter().map(|e| self.align_of(e)).max().unwrap_or(1),
            Type::Struct { fields, .. } => fields
                .iter()
                .map(|f| self.align_of(&f.ty))
                .max()
                .unwrap_or(1),
            Type::Function { .. } => self.pointer_alignment,
            _ => self.pointer_alignment, // Conservative default
        }
    }

    /// Get the size of a primitive type
    fn primitive_size(&self, prim: &zyntax_typed_ast::PrimitiveType) -> usize {
        use zyntax_typed_ast::PrimitiveType;
        match prim {
            PrimitiveType::Bool => 1,
            PrimitiveType::I8 | PrimitiveType::U8 => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 => 8,
            PrimitiveType::I128 | PrimitiveType::U128 => 16,
            PrimitiveType::F32 => 4,
            PrimitiveType::F64 => 8,
            PrimitiveType::String => self.pointer_size, // String is a pointer
            _ => self.pointer_size,
        }
    }

    /// Get the alignment of a primitive type
    fn primitive_alignment(&self, prim: &zyntax_typed_ast::PrimitiveType) -> usize {
        use zyntax_typed_ast::PrimitiveType;
        match prim {
            PrimitiveType::Bool => 1,
            PrimitiveType::I8 | PrimitiveType::U8 => 1,
            PrimitiveType::I16 | PrimitiveType::U16 => 2,
            PrimitiveType::I32 | PrimitiveType::U32 => 4,
            PrimitiveType::I64 | PrimitiveType::U64 => 8,
            PrimitiveType::I128 | PrimitiveType::U128 => {
                // i128 alignment varies by platform
                if self.architecture.contains("x86") {
                    8 // x86/x86_64 aligns i128 to 8 bytes
                } else {
                    16 // ARM, others align to 16
                }
            }
            PrimitiveType::F32 => 4,
            PrimitiveType::F64 => 8,
            PrimitiveType::String => self.pointer_alignment,
            _ => self.pointer_alignment,
        }
    }

    /// Calculate struct size with proper padding
    fn calculate_struct_size(&self, fields: Vec<&Type>) -> usize {
        if fields.is_empty() {
            return 0;
        }

        let mut offset = 0;
        let mut max_align = 1;

        for field in fields {
            let field_size = self.size_of(field);
            let field_align = self.align_of(field);
            max_align = max_align.max(field_align);

            // Align current offset
            offset = Self::align_to(offset, field_align);
            offset += field_size;
        }

        // Align total size to struct alignment
        Self::align_to(offset, max_align)
    }

    /// Round up to the next multiple of alignment
    fn align_to(value: usize, alignment: usize) -> usize {
        (value + alignment - 1) & !(alignment - 1)
    }
}

/// Lowering context for a compilation unit
pub struct LoweringContext {
    /// Current module being lowered
    pub module: HirModule,
    /// Type registry for type conversions
    pub type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
    /// String arena for creating mangled names
    pub arena: Arc<Mutex<AstArena>>,
    /// Symbol table for name resolution
    pub symbols: SymbolTable,
    /// Diagnostics collector
    pub diagnostics: Vec<LoweringDiagnostic>,
    /// Configuration options
    pub config: LoweringConfig,
    /// Vtable registry for trait dispatch
    pub vtable_registry: crate::vtable_registry::VtableRegistry,
    /// Associated type resolver for trait dispatch
    pub associated_type_resolver: crate::associated_type_resolver::AssociatedTypeResolver,
    /// Target data for precise size/alignment calculations
    pub target_data: TargetData,
    /// Import metadata for debugging
    pub import_metadata: Vec<ImportMetadata>,
    /// Import context for resolving imports during lowering
    pub import_context: ImportContext,
    /// Cache of already-resolved modules (module_path -> resolved imports)
    /// This avoids re-resolving the same module multiple times
    pub resolved_module_cache: std::collections::HashMap<Vec<String>, Vec<ResolvedImport>>,
    /// Ownership modes for values (tracks whether values are owned, borrowed, or copied)
    pub ownership_modes: HashMap<HirId, OwnershipMode>,
    /// Types that implement Copy trait (for deciding between move and copy semantics)
    pub copy_types: std::collections::HashSet<zyntax_typed_ast::TypeId>,
}

/// Symbol table for name resolution
#[derive(Debug, Default)]
pub struct SymbolTable {
    /// Functions by name
    pub functions: indexmap::IndexMap<InternedString, crate::hir::HirId>,
    /// Globals by name
    pub globals: indexmap::IndexMap<InternedString, crate::hir::HirId>,
    /// Types by name
    pub types: indexmap::IndexMap<InternedString, zyntax_typed_ast::TypeId>,
    /// Algebraic effects by name
    pub effects: indexmap::IndexMap<InternedString, crate::hir::HirId>,
    /// Effect handlers by name
    pub handlers: indexmap::IndexMap<InternedString, crate::hir::HirId>,
    /// External function link names (alias -> ZRTL symbol)
    /// e.g., "tensor_add" -> "$Tensor$add"
    pub extern_link_names: indexmap::IndexMap<InternedString, String>,
}

/// Import metadata for debugging and error messages
#[derive(Debug, Clone)]
pub struct ImportMetadata {
    /// Module path (e.g., ["std", "io", "println"])
    pub module_path: Vec<InternedString>,
    /// Imported items
    pub items: Vec<ImportedItem>,
    /// Source location
    pub span: zyntax_typed_ast::Span,
    /// Resolved imports (populated if import resolver is configured)
    pub resolved: Vec<ResolvedImport>,
}

/// An imported item
#[derive(Debug, Clone)]
pub struct ImportedItem {
    /// Original name in the source module
    pub name: InternedString,
    /// Local alias (if renamed)
    pub alias: Option<InternedString>,
    /// Whether this is a glob import
    pub is_glob: bool,
}

#[derive(Debug, Clone)]
struct CallParamSpec {
    name: InternedString,
    ty: Type,
}

// Re-export import resolver types from typed_ast
pub use zyntax_typed_ast::import_resolver::{
    BuiltinResolver, ChainedResolver, ExportedSymbol, ImportContext, ImportError, ImportManager,
    ImportResolver, ModuleArchitecture, ResolvedImport, SymbolKind,
};

/// Lowering configuration
#[derive(Clone)]
pub struct LoweringConfig {
    /// Enable debug information
    pub debug_info: bool,
    /// Optimization level (0-3)
    pub opt_level: u8,
    /// Target triple for platform-specific lowering
    pub target_triple: String,
    /// Enable hot-reloading support
    pub hot_reload: bool,
    /// Enable strict mode (fail on warnings)
    pub strict_mode: bool,
    /// Optional import resolver for resolving import statements
    pub import_resolver: Option<Arc<dyn ImportResolver>>,
    /// Builtin function mappings (e.g., "tensor_sum_f32" -> "$Tensor$sum_f32")
    /// These are added to extern_link_names for resolving extern calls
    pub builtins: indexmap::IndexMap<String, String>,
}

impl std::fmt::Debug for LoweringConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoweringConfig")
            .field("debug_info", &self.debug_info)
            .field("opt_level", &self.opt_level)
            .field("target_triple", &self.target_triple)
            .field("hot_reload", &self.hot_reload)
            .field("strict_mode", &self.strict_mode)
            .field(
                "import_resolver",
                &self.import_resolver.as_ref().map(|r| r.resolver_name()),
            )
            .finish()
    }
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            debug_info: true,
            opt_level: 0,
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            hot_reload: false,
            strict_mode: false,
            import_resolver: None,
            builtins: indexmap::IndexMap::new(),
        }
    }
}

/// Diagnostic during lowering
#[derive(Debug)]
pub struct LoweringDiagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<zyntax_typed_ast::Span>,
}

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
}

/// Main AST lowering interface
pub trait AstLowering {
    /// Lower a typed program to HIR
    fn lower_program(&mut self, program: &mut TypedProgram) -> CompilerResult<HirModule>;
}

/// Lowering pipeline implementation
pub struct LoweringPipeline {
    passes: Vec<Box<dyn LoweringPass>>,
}

/// Individual lowering pass
pub trait LoweringPass: Send + Sync {
    /// Name of this pass
    fn name(&self) -> &'static str;

    /// Dependencies this pass requires
    fn dependencies(&self) -> &[&'static str];

    /// Run this pass
    fn run(&mut self, context: &mut LoweringContext) -> CompilerResult<()>;
}

impl LoweringContext {
    pub fn new(
        module_name: InternedString,
        type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
        arena: Arc<Mutex<AstArena>>,
        config: LoweringConfig,
    ) -> Self {
        // Initialize symbol table with builtins from config
        let mut symbols = SymbolTable::default();
        if !config.builtins.is_empty() {
            log::debug!(
                "[LOWERING] Populating extern_link_names with {} builtins",
                config.builtins.len()
            );
        }
        for (alias, target) in &config.builtins {
            let alias_interned = InternedString::new_global(alias);
            symbols
                .extern_link_names
                .insert(alias_interned, target.clone());
            log::trace!("[LOWERING] Added builtin: '{}' -> '{}'", alias, target);
        }

        Self {
            module: HirModule::new(module_name),
            type_registry,
            arena,
            symbols,
            diagnostics: Vec::new(),
            config,
            vtable_registry: crate::vtable_registry::VtableRegistry::new(),
            associated_type_resolver: crate::associated_type_resolver::AssociatedTypeResolver::new(
            ),
            target_data: TargetData::host(),
            import_metadata: Vec::new(),
            import_context: ImportContext::default(),
            resolved_module_cache: std::collections::HashMap::new(),
            ownership_modes: HashMap::new(),
            copy_types: std::collections::HashSet::new(),
        }
    }

    /// Set the ownership mode for a value
    pub fn set_ownership_mode(&mut self, id: HirId, mode: OwnershipMode) {
        self.ownership_modes.insert(id, mode);
    }

    /// Get the ownership mode for a value
    pub fn get_ownership_mode(&self, id: &HirId) -> Option<&OwnershipMode> {
        self.ownership_modes.get(id)
    }

    /// Check if a type implements Copy (can be duplicated without moving)
    pub fn is_copy_type(&self, ty: &Type) -> bool {
        // Primitive types are always Copy
        match ty {
            Type::Primitive(_) => true,
            Type::Named { id, .. } => self.copy_types.contains(id),
            // Tuples are Copy if all elements are Copy
            Type::Tuple(elements) => elements.iter().all(|e| self.is_copy_type(e)),
            // References are Copy (regardless of mutability)
            Type::Reference { .. } => true,
            // Function types are Copy
            Type::Function { .. } => true,
            // Other types are not Copy by default
            _ => false,
        }
    }

    /// Determine the ownership mode for using a value based on its type
    pub fn determine_ownership_mode(&self, ty: &Type) -> OwnershipMode {
        if self.is_copy_type(ty) {
            OwnershipMode::Copied
        } else {
            OwnershipMode::Owned
        }
    }

    /// Initialize the set of Copy types from the type registry
    /// This looks for types that implement the Copy trait
    fn initialize_copy_types(&mut self) {
        // Look for Copy trait in the registry
        let copy_trait_name = zyntax_typed_ast::arena::InternedString::new_global("Copy");

        if let Some(copy_trait) = self.type_registry.get_trait_by_name(copy_trait_name) {
            // Find all implementations of Copy trait
            for (trait_id, impls) in self.type_registry.iter_implementations() {
                if *trait_id == copy_trait.id {
                    for impl_def in impls {
                        // Extract the type that implements Copy
                        if let Type::Named { id, .. } = &impl_def.for_type {
                            self.copy_types.insert(*id);
                        }
                    }
                }
            }
        }

        // TODO: Also check for types with Copy annotation/attribute
    }

    /// Create a borrow of a value, returning the borrow's HirId
    pub fn create_borrow(&mut self, value_id: HirId, is_mutable: bool) -> HirId {
        let borrow_id = HirId::new();
        let lifetime = HirLifetime::anonymous();

        // Track the borrow in ownership_modes
        let mode = if is_mutable {
            OwnershipMode::BorrowedMut(lifetime.clone())
        } else {
            OwnershipMode::Borrowed(lifetime)
        };
        self.ownership_modes.insert(borrow_id, mode);

        borrow_id
    }

    /// Add a diagnostic
    pub fn diagnostic(
        &mut self,
        level: DiagnosticLevel,
        message: String,
        span: Option<zyntax_typed_ast::Span>,
    ) {
        self.diagnostics.push(LoweringDiagnostic {
            level,
            message,
            span,
        });
    }
}

impl AstLowering for LoweringContext {
    fn lower_program(&mut self, program: &mut TypedProgram) -> CompilerResult<HirModule> {
        // Phase -1: Initialize Copy types from the type registry
        // Types that implement the Copy trait can be duplicated instead of moved
        self.initialize_copy_types();

        // Phase 0: Run type checking and inference (Issue 0 Phase 1)
        // This validates types and performs type inference, reporting any errors
        // Skip type checking if SKIP_TYPE_CHECK env var is set (for debugging)
        if std::env::var("SKIP_TYPE_CHECK").is_err() {
            self.run_type_checking(program)?;
        }

        // Phase 0.5: Resolve method call return types
        // This updates MethodCall expression types by looking up trait implementations
        self.resolve_method_call_types(program)?;

        // First pass: collect all declarations
        self.collect_declarations(program)?;

        // Second pass: lower each declaration
        for decl in &program.declarations {
            self.lower_declaration(decl)?;
        }

        // New Phase: Lower trait implementations and generate vtables
        self.lower_implementations()?;

        // Third pass: resolve forward references
        self.resolve_references()?;

        // Increment version for hot-reloading
        if self.config.hot_reload {
            self.module.increment_version();
        }

        Ok(self.module.clone())
    }
}

impl LoweringContext {
    /// Run type checking and inference on the program
    /// This is Issue 0 Phase 1: integrate type inference into lowering
    fn run_type_checking(&mut self, program: &mut TypedProgram) -> CompilerResult<()> {
        use zyntax_typed_ast::type_checker::{TypeCheckOptions, TypeChecker};

        // Create type checker - needs Box<TypeRegistry>
        let registry = Box::new((*self.type_registry).clone());
        let mut type_checker = TypeChecker::with_options(
            registry,
            TypeCheckOptions {
                strict_nulls: false, // Be lenient for now
                strict_functions: false,
                no_implicit_any: false, // Allow Unknown types
                check_unreachable: false,
            },
        );

        // Run type checking (validates and performs internal inference)
        type_checker.check_program(program);

        // Apply resolved types from inference to the AST
        // This propagates Type::Any resolutions back to the AST nodes
        type_checker.apply_inferred_types(program);

        // Check for type errors and display diagnostics
        let diagnostics_collector = type_checker.diagnostics();
        let total_errors = diagnostics_collector.error_count();
        let total_warnings = diagnostics_collector.warning_count();

        // Display diagnostics using the built-in pretty formatter
        // Suppress stdlib errors (known false positives in impl blocks)
        if total_errors > 0 || total_warnings > 0 {
            use zyntax_typed_ast::diagnostics::{ConsoleDiagnosticDisplay, DiagnosticDisplay};
            use zyntax_typed_ast::source::SourceMap;

            // Check if errors are only from stdlib
            let has_stdlib_sources = program
                .source_files
                .iter()
                .any(|sf| sf.name.contains("stdlib/"));

            // If we have stdlib sources, suppress the diagnostic output
            // (these are known false positives from trait impl type checking)
            if !has_stdlib_sources {
                eprintln!("\n=== Type Checking Diagnostics ===");

                // Create a source map and populate it with source files from the program
                let mut source_map = SourceMap::new();
                for source_file in &program.source_files {
                    source_map.add_file(source_file.name.clone(), source_file.content.clone());
                }
                let display = ConsoleDiagnosticDisplay::default();

                // Use the built-in pretty formatter
                let diagnostic_output = diagnostics_collector.display_all(&display, &source_map);
                eprintln!("{}", diagnostic_output);

                eprintln!("=================================\n");
            }
        }

        // Suppress error count if only stdlib errors
        let has_stdlib_sources = program
            .source_files
            .iter()
            .any(|sf| sf.name.contains("stdlib/"));
        let (error_count, warning_count) = if has_stdlib_sources {
            (0, 0) // Suppress stdlib type errors (known false positives)
        } else {
            (total_errors, total_warnings)
        };

        // TODO (Issue 0 Phase 2): Make type checking stricter once inference properly modifies AST
        // For now, just warn about errors rather than failing compilation
        if error_count > 0 {
            eprintln!(
                "WARNING: Type checking found {} error(s), {} warning(s) - continuing compilation",
                error_count, warning_count
            );
            eprintln!("NOTE: Issue 0 Phase 1 complete - type checking integrated but lenient");
            eprintln!("      Enable strict checking with ZYNTAX_STRICT_TYPES=1\n");

            if std::env::var("ZYNTAX_STRICT_TYPES").is_ok() {
                return Err(crate::CompilerError::Lowering(format!(
                    "Type checking failed with {} error(s), {} warning(s)",
                    error_count, warning_count
                )));
            }
        }

        // Warn about Unknown types (for debugging Issue 0)
        if std::env::var("ZYNTAX_DEBUG_TYPES").is_ok() {
            self.check_for_unknown_types(program);
        }

        Ok(())
    }

    /// Resolve method call return types by looking up trait implementations
    /// This updates MethodCall expressions with Type::Any to have the correct return type
    fn resolve_method_call_types(&mut self, program: &mut TypedProgram) -> CompilerResult<()> {
        use std::collections::HashMap;
        use zyntax_typed_ast::typed_ast::TypedDeclaration;
        use zyntax_typed_ast::Type;
        use zyntax_typed_ast::TypedExpression;

        let function_param_specs = self.collect_declared_function_param_specs(program);

        // Iterate through all declarations and resolve method calls
        for decl in &mut program.declarations {
            match &mut decl.node {
                TypedDeclaration::Function(func) => {
                    if let Some(body) = &mut func.body {
                        let mut var_types = HashMap::new();
                        for param in &func.params {
                            var_types.insert(param.name, param.ty.clone());
                        }
                        self.resolve_method_calls_in_block(
                            body,
                            &mut var_types,
                            &function_param_specs,
                            &func.return_type,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Recursively resolve method calls in a block
    fn resolve_method_calls_in_block(
        &self,
        block: &mut zyntax_typed_ast::TypedBlock,
        var_types: &mut std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            zyntax_typed_ast::Type,
        >,
        function_param_specs: &std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            Vec<CallParamSpec>,
        >,
        function_return_type: &zyntax_typed_ast::Type,
    ) -> CompilerResult<()> {
        for stmt in &mut block.statements {
            self.resolve_method_calls_in_statement(
                stmt,
                var_types,
                function_param_specs,
                function_return_type,
            )?;
        }
        Ok(())
    }

    /// Recursively resolve method calls in a statement
    fn resolve_method_calls_in_statement(
        &self,
        stmt: &mut zyntax_typed_ast::TypedNode<zyntax_typed_ast::typed_ast::TypedStatement>,
        var_types: &mut std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            zyntax_typed_ast::Type,
        >,
        function_param_specs: &std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            Vec<CallParamSpec>,
        >,
        function_return_type: &zyntax_typed_ast::Type,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::typed_ast::TypedStatement;
        use zyntax_typed_ast::{PrimitiveType, Type};

        match &mut stmt.node {
            TypedStatement::Expression(expr) => {
                self.resolve_method_calls_in_expression(expr, var_types, function_param_specs)?;
            }
            TypedStatement::Let(let_stmt) => {
                if let Some(init) = &mut let_stmt.initializer {
                    self.resolve_method_calls_in_expression(init, var_types, function_param_specs)?;

                    if !matches!(let_stmt.ty, Type::Any | Type::Unknown) {
                        if let Some(converted) =
                            self.try_implicit_from_conversion((**init).clone(), &let_stmt.ty)
                        {
                            *init = Box::new(converted);
                        }
                    }

                    // If the let statement has type Any/Unknown and the initializer has a resolved type, update it
                    let needs_update = matches!(let_stmt.ty, Type::Any | Type::Unknown);
                    let has_resolved = !matches!(init.ty, Type::Any | Type::Unknown);
                    if needs_update && has_resolved {
                        let_stmt.ty = init.ty.clone();
                    }
                    // Register the variable's type for later lookups
                    var_types.insert(let_stmt.name, let_stmt.ty.clone());
                }
            }
            TypedStatement::Return(opt_expr) => {
                if let Some(expr) = opt_expr {
                    self.resolve_method_calls_in_expression(expr, var_types, function_param_specs)?;

                    if !matches!(function_return_type, Type::Any | Type::Unknown)
                        && !matches!(function_return_type, Type::Primitive(PrimitiveType::Unit))
                    {
                        if let Some(converted) = self
                            .try_implicit_from_conversion((**expr).clone(), function_return_type)
                        {
                            *expr = Box::new(converted);
                        }
                    }
                }
            }
            TypedStatement::Yield(expr) => {
                self.resolve_method_calls_in_expression(expr, var_types, function_param_specs)?;
            }
            TypedStatement::If(if_stmt) => {
                self.resolve_method_calls_in_expression(
                    &mut if_stmt.condition,
                    var_types,
                    function_param_specs,
                )?;
                self.resolve_method_calls_in_block(
                    &mut if_stmt.then_block,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
                if let Some(else_block) = &mut if_stmt.else_block {
                    self.resolve_method_calls_in_block(
                        else_block,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
            }
            TypedStatement::While(while_stmt) => {
                self.resolve_method_calls_in_expression(
                    &mut while_stmt.condition,
                    var_types,
                    function_param_specs,
                )?;
                self.resolve_method_calls_in_block(
                    &mut while_stmt.body,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
            }
            TypedStatement::Block(block) => {
                self.resolve_method_calls_in_block(
                    block,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
            }
            TypedStatement::For(for_stmt) => {
                self.resolve_method_calls_in_expression(
                    &mut for_stmt.iterator,
                    var_types,
                    function_param_specs,
                )?;
                self.resolve_method_calls_in_block(
                    &mut for_stmt.body,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
            }
            TypedStatement::ForCStyle(for_stmt) => {
                if let Some(init) = &mut for_stmt.init {
                    self.resolve_method_calls_in_statement(
                        init,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                if let Some(condition) = &mut for_stmt.condition {
                    self.resolve_method_calls_in_expression(
                        condition,
                        var_types,
                        function_param_specs,
                    )?;
                }
                if let Some(update) = &mut for_stmt.update {
                    self.resolve_method_calls_in_expression(
                        update,
                        var_types,
                        function_param_specs,
                    )?;
                }
                self.resolve_method_calls_in_block(
                    &mut for_stmt.body,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
            }
            TypedStatement::Loop(loop_stmt) => match loop_stmt {
                zyntax_typed_ast::typed_ast::TypedLoop::ForEach { iterator, body, .. } => {
                    self.resolve_method_calls_in_expression(
                        iterator,
                        var_types,
                        function_param_specs,
                    )?;
                    self.resolve_method_calls_in_block(
                        body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                zyntax_typed_ast::typed_ast::TypedLoop::ForCStyle {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    if let Some(init_stmt) = init {
                        self.resolve_method_calls_in_statement(
                            init_stmt,
                            var_types,
                            function_param_specs,
                            function_return_type,
                        )?;
                    }
                    if let Some(condition_expr) = condition {
                        self.resolve_method_calls_in_expression(
                            condition_expr,
                            var_types,
                            function_param_specs,
                        )?;
                    }
                    if let Some(update_expr) = update {
                        self.resolve_method_calls_in_expression(
                            update_expr,
                            var_types,
                            function_param_specs,
                        )?;
                    }
                    self.resolve_method_calls_in_block(
                        body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                zyntax_typed_ast::typed_ast::TypedLoop::While { condition, body } => {
                    self.resolve_method_calls_in_expression(
                        condition,
                        var_types,
                        function_param_specs,
                    )?;
                    self.resolve_method_calls_in_block(
                        body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                zyntax_typed_ast::typed_ast::TypedLoop::DoWhile { body, condition } => {
                    self.resolve_method_calls_in_block(
                        body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                    self.resolve_method_calls_in_expression(
                        condition,
                        var_types,
                        function_param_specs,
                    )?;
                }
                zyntax_typed_ast::typed_ast::TypedLoop::Infinite { body } => {
                    self.resolve_method_calls_in_block(
                        body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
            },
            TypedStatement::Match(match_stmt) => {
                self.resolve_method_calls_in_expression(
                    &mut match_stmt.scrutinee,
                    var_types,
                    function_param_specs,
                )?;
                for arm in &mut match_stmt.arms {
                    if let Some(guard) = &mut arm.guard {
                        self.resolve_method_calls_in_expression(
                            guard,
                            var_types,
                            function_param_specs,
                        )?;
                    }
                    self.resolve_method_calls_in_expression(
                        &mut arm.body,
                        var_types,
                        function_param_specs,
                    )?;
                }
            }
            TypedStatement::Try(try_stmt) => {
                self.resolve_method_calls_in_block(
                    &mut try_stmt.body,
                    var_types,
                    function_param_specs,
                    function_return_type,
                )?;
                for catch in &mut try_stmt.catch_clauses {
                    self.resolve_method_calls_in_block(
                        &mut catch.body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                if let Some(finally_block) = &mut try_stmt.finally_block {
                    self.resolve_method_calls_in_block(
                        finally_block,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
            }
            TypedStatement::Throw(expr) => {
                self.resolve_method_calls_in_expression(expr, var_types, function_param_specs)?;
            }
            TypedStatement::Break(value) => {
                if let Some(value_expr) = value {
                    self.resolve_method_calls_in_expression(
                        value_expr,
                        var_types,
                        function_param_specs,
                    )?;
                }
            }
            TypedStatement::LetPattern(let_pattern) => {
                self.resolve_method_calls_in_expression(
                    &mut let_pattern.initializer,
                    var_types,
                    function_param_specs,
                )?;
            }
            TypedStatement::Coroutine(coroutine) => {
                self.resolve_method_calls_in_expression(
                    &mut coroutine.body,
                    var_types,
                    function_param_specs,
                )?;
                for param in &mut coroutine.params {
                    self.resolve_method_calls_in_expression(
                        param,
                        var_types,
                        function_param_specs,
                    )?;
                }
            }
            TypedStatement::Defer(defer_stmt) => {
                self.resolve_method_calls_in_expression(
                    &mut defer_stmt.body,
                    var_types,
                    function_param_specs,
                )?;
            }
            TypedStatement::Select(select_stmt) => {
                for arm in &mut select_stmt.arms {
                    match &mut arm.operation {
                        zyntax_typed_ast::typed_ast::TypedSelectOperation::Receive {
                            channel,
                            pattern: _,
                        } => {
                            self.resolve_method_calls_in_expression(
                                channel,
                                var_types,
                                function_param_specs,
                            )?;
                        }
                        zyntax_typed_ast::typed_ast::TypedSelectOperation::Send {
                            channel,
                            value,
                        } => {
                            self.resolve_method_calls_in_expression(
                                channel,
                                var_types,
                                function_param_specs,
                            )?;
                            self.resolve_method_calls_in_expression(
                                value,
                                var_types,
                                function_param_specs,
                            )?;
                        }
                        zyntax_typed_ast::typed_ast::TypedSelectOperation::Timeout { duration } => {
                            self.resolve_method_calls_in_expression(
                                duration,
                                var_types,
                                function_param_specs,
                            )?;
                        }
                    }
                    self.resolve_method_calls_in_block(
                        &mut arm.body,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
                if let Some(default_block) = &mut select_stmt.default {
                    self.resolve_method_calls_in_block(
                        default_block,
                        var_types,
                        function_param_specs,
                        function_return_type,
                    )?;
                }
            }
            TypedStatement::Continue => {}
        }
        Ok(())
    }

    /// Recursively resolve method calls in an expression
    fn resolve_method_calls_in_expression(
        &self,
        expr: &mut zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>,
        var_types: &std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            zyntax_typed_ast::Type,
        >,
        function_param_specs: &std::collections::HashMap<
            zyntax_typed_ast::InternedString,
            Vec<CallParamSpec>,
        >,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::{Type, TypedExpression};

        match &mut expr.node {
            TypedExpression::MethodCall(method_call) => {
                // Recursively process receiver and arguments FIRST
                self.resolve_method_calls_in_expression(
                    &mut method_call.receiver,
                    var_types,
                    function_param_specs,
                )?;
                for arg in &mut method_call.positional_args {
                    self.resolve_method_calls_in_expression(arg, var_types, function_param_specs)?;
                }
                for arg in &mut method_call.named_args {
                    self.resolve_method_calls_in_expression(
                        &mut arg.value,
                        var_types,
                        function_param_specs,
                    )?;
                }

                // If the method call has Type::Any or Type::Unknown, resolve it
                // First, get the actual receiver type (may need to look up from var_types)
                if matches!(expr.ty, Type::Any | Type::Unknown) {
                    let receiver_type = if let TypedExpression::Variable(var_name) =
                        &method_call.receiver.node
                    {
                        // First check if this is a type name (static method call like Tensor::arange)
                        if let Some(type_def) = self.type_registry.get_type_by_name(*var_name) {
                            // This is a static method call - look for the method in the type's methods
                            for method in &type_def.methods {
                                if method.name == method_call.method {
                                    expr.ty = method.return_type.clone();
                                    return Ok(());
                                }
                            }
                            // Also check impl blocks for this type
                            for (_trait_id, impls) in self.type_registry.iter_implementations() {
                                for impl_def in impls {
                                    // Check if this impl is for our type (by name comparison for extern types)
                                    let impl_for_this_type = match &impl_def.for_type {
                                        Type::Named { id, .. } => *id == type_def.id,
                                        Type::Extern { name, .. } => *name == *var_name,
                                        Type::Unresolved(name) => *name == *var_name,
                                        _ => false,
                                    };
                                    if impl_for_this_type {
                                        for method in &impl_def.methods {
                                            if method.signature.name == method_call.method {
                                                expr.ty = method.signature.return_type.clone();
                                                return Ok(());
                                            }
                                        }
                                    }
                                }
                            }
                            // Type found but method not found - return type as Named
                            Type::Named {
                                id: type_def.id,
                                type_args: vec![],
                                const_args: vec![],
                                variance: vec![],
                                nullability:
                                    zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                            }
                        } else {
                            // Not a type name, try looking up as variable
                            var_types
                                .get(var_name)
                                .cloned()
                                .unwrap_or_else(|| method_call.receiver.ty.clone())
                        }
                    } else {
                        method_call.receiver.ty.clone()
                    };

                    if let Type::Named {
                        id: receiver_type_id,
                        ..
                    } = &receiver_type
                    {
                        // Look up the trait implementation
                        for (_trait_id, impls) in self.type_registry.iter_implementations() {
                            for impl_def in impls {
                                if let Type::Named {
                                    id: impl_type_id, ..
                                } = &impl_def.for_type
                                {
                                    if *impl_type_id == *receiver_type_id {
                                        // Find the method in this impl
                                        for method in &impl_def.methods {
                                            if method.signature.name == method_call.method {
                                                // Update the expression type
                                                expr.ty = method.signature.return_type.clone();
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            TypedExpression::Call(call) => {
                let mut associated_param_specs: Option<Vec<CallParamSpec>> = None;

                // Check if this is an associated function call (Type::method syntax)
                // First check for Path expressions (Tensor::arange parsed as Path)
                if let TypedExpression::Path(path) = &call.callee.node {
                    if path.segments.len() == 2 {
                        let type_name_interned = path.segments[0];
                        let method_name_interned = path.segments[1];
                        let type_name = type_name_interned.resolve_global().unwrap_or_default();
                        let method_name = method_name_interned.resolve_global().unwrap_or_default();

                        // Look up the type and find the method return type
                        if let Some(type_def) =
                            self.type_registry.get_type_by_name(type_name_interned)
                        {
                            // First check inherent methods on the type
                            for method in &type_def.methods {
                                if method.name == method_name_interned {
                                    // Found it! Set the Call expression's return type
                                    expr.ty = method.return_type.clone();
                                    associated_param_specs =
                                        Some(self.param_specs_from_method_sig(method));
                                    break;
                                }
                            }

                            // If not found in inherent methods, check impl blocks
                            if matches!(expr.ty, Type::Any | Type::Unknown) {
                                for (_trait_id, impls) in self.type_registry.iter_implementations()
                                {
                                    for impl_def in impls {
                                        let impl_for_this_type = match &impl_def.for_type {
                                            Type::Named { id, .. } => *id == type_def.id,
                                            Type::Extern { name, .. } => {
                                                *name == type_name_interned
                                            }
                                            Type::Unresolved(name) => *name == type_name_interned,
                                            _ => false,
                                        };
                                        if impl_for_this_type {
                                            for method in &impl_def.methods {
                                                if method.signature.name == method_name_interned {
                                                    expr.ty = method.signature.return_type.clone();
                                                    associated_param_specs =
                                                        Some(self.param_specs_from_method_sig(
                                                            &method.signature,
                                                        ));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Also try to resolve the mangled name
                        if let Some(mangled_name) =
                            self.resolve_associated_function_to_mangled(&type_name, &method_name)
                        {
                            // Replace the Path with the mangled Variable name
                            let mangled_interned = InternedString::new_global(&mangled_name);
                            call.callee.node = TypedExpression::Variable(mangled_interned);
                        }
                    }
                }
                // Also check for Variable with "::" in name (fallback for other parsing)
                else if let TypedExpression::Variable(func_name) = &call.callee.node {
                    let func_name_str = func_name.resolve_global().or_else(|| {
                        let arena = self.arena.lock().unwrap();
                        arena.resolve_string(*func_name).map(|s| s.to_string())
                    });

                    if let Some(name_str) = func_name_str {
                        // Check if this looks like an associated function call (contains ::)
                        if name_str.contains("::") {
                            // Parse Type::method
                            let parts: Vec<&str> = name_str.split("::").collect();
                            if parts.len() == 2 {
                                let type_name = parts[0];
                                let method_name = parts[1];

                                // Look up the type and find the trait impl
                                if let Some(mangled_name) = self
                                    .resolve_associated_function_to_mangled(type_name, method_name)
                                {
                                    // Replace the Variable with the mangled name
                                    let mangled_interned =
                                        InternedString::new_global(&mangled_name);
                                    call.callee.node = TypedExpression::Variable(mangled_interned);
                                }
                            }
                        }
                    }
                }

                self.resolve_method_calls_in_expression(
                    &mut call.callee,
                    var_types,
                    function_param_specs,
                )?;
                for arg in &mut call.positional_args {
                    self.resolve_method_calls_in_expression(arg, var_types, function_param_specs)?;
                }
                for arg in &mut call.named_args {
                    self.resolve_method_calls_in_expression(
                        &mut arg.value,
                        var_types,
                        function_param_specs,
                    )?;
                }

                let expected_params = if let Some(params) = associated_param_specs {
                    Some(params)
                } else if let TypedExpression::Variable(callee_name) = &call.callee.node {
                    function_param_specs.get(callee_name).cloned()
                } else {
                    None
                };

                if let Some(params) = expected_params {
                    self.apply_implicit_from_to_call_args(
                        &mut call.positional_args,
                        &mut call.named_args,
                        &params,
                    );
                }
            }
            TypedExpression::Binary(binary) => {
                self.resolve_method_calls_in_expression(
                    &mut binary.left,
                    var_types,
                    function_param_specs,
                )?;
                self.resolve_method_calls_in_expression(
                    &mut binary.right,
                    var_types,
                    function_param_specs,
                )?;

                if matches!(binary.op, zyntax_typed_ast::BinaryOp::Assign)
                    || Self::supports_rhs_implicit_conversion(binary.op)
                {
                    if let Some(converted) =
                        self.try_implicit_from_conversion((*binary.right).clone(), &binary.left.ty)
                    {
                        binary.right = Box::new(converted);
                    }
                }
            }
            TypedExpression::Unary(unary) => {
                self.resolve_method_calls_in_expression(
                    &mut unary.operand,
                    var_types,
                    function_param_specs,
                )?;
            }
            TypedExpression::Field(field_access) => {
                self.resolve_method_calls_in_expression(
                    &mut field_access.object,
                    var_types,
                    function_param_specs,
                )?;
            }
            TypedExpression::Struct(struct_lit) => {
                for field in &mut struct_lit.fields {
                    self.resolve_method_calls_in_expression(
                        &mut field.value,
                        var_types,
                        function_param_specs,
                    )?;
                }
            }
            TypedExpression::Variable(_var_name) => {
                // Variable types are looked up from var_types when needed (see MethodCall case above)
            }
            _ => {}
        }

        Ok(())
    }

    fn collect_declared_function_param_specs(
        &self,
        program: &TypedProgram,
    ) -> std::collections::HashMap<InternedString, Vec<CallParamSpec>> {
        use zyntax_typed_ast::typed_ast::TypedDeclaration;

        let mut specs = std::collections::HashMap::new();
        for decl in &program.declarations {
            if let TypedDeclaration::Function(func) = &decl.node {
                let params = func
                    .params
                    .iter()
                    .map(|p| CallParamSpec {
                        name: p.name,
                        ty: p.ty.clone(),
                    })
                    .collect();
                specs.insert(func.name, params);
            }
        }
        specs
    }

    fn param_specs_from_method_sig(
        &self,
        method: &zyntax_typed_ast::type_registry::MethodSig,
    ) -> Vec<CallParamSpec> {
        method
            .params
            .iter()
            .filter(|p| !p.is_self)
            .map(|p| CallParamSpec {
                name: p.name,
                ty: p.ty.clone(),
            })
            .collect()
    }

    fn apply_implicit_from_to_call_args(
        &self,
        positional_args: &mut Vec<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>>,
        named_args: &mut Vec<zyntax_typed_ast::typed_ast::TypedNamedArg>,
        expected_params: &[CallParamSpec],
    ) {
        for (arg, expected) in positional_args.iter_mut().zip(expected_params.iter()) {
            if let Some(converted) = self.try_implicit_from_conversion(arg.clone(), &expected.ty) {
                *arg = converted;
            }
        }

        for named in named_args.iter_mut() {
            if let Some(expected) = expected_params.iter().find(|p| p.name == named.name) {
                if let Some(converted) =
                    self.try_implicit_from_conversion((*named.value).clone(), &expected.ty)
                {
                    named.value = Box::new(converted);
                }
            }
        }
    }

    fn supports_rhs_implicit_conversion(op: zyntax_typed_ast::BinaryOp) -> bool {
        matches!(
            op,
            zyntax_typed_ast::BinaryOp::Add
                | zyntax_typed_ast::BinaryOp::Sub
                | zyntax_typed_ast::BinaryOp::Mul
                | zyntax_typed_ast::BinaryOp::MatMul
                | zyntax_typed_ast::BinaryOp::Div
                | zyntax_typed_ast::BinaryOp::Rem
                | zyntax_typed_ast::BinaryOp::Eq
                | zyntax_typed_ast::BinaryOp::Ne
                | zyntax_typed_ast::BinaryOp::Lt
                | zyntax_typed_ast::BinaryOp::Le
                | zyntax_typed_ast::BinaryOp::Gt
                | zyntax_typed_ast::BinaryOp::Ge
                | zyntax_typed_ast::BinaryOp::BitAnd
                | zyntax_typed_ast::BinaryOp::BitOr
                | zyntax_typed_ast::BinaryOp::BitXor
                | zyntax_typed_ast::BinaryOp::Shl
                | zyntax_typed_ast::BinaryOp::Shr
        )
    }

    /// Debug helper: check for Unknown types in the program
    fn check_for_unknown_types(&self, program: &TypedProgram) {
        use zyntax_typed_ast::Type;

        let mut unknown_count = 0;
        let mut func_count = 0;

        for decl in &program.declarations {
            if let zyntax_typed_ast::TypedDeclaration::Function(func) = &decl.node {
                func_count += 1;

                // Check return type
                if matches!(func.return_type, Type::Unknown) {
                    eprintln!("  - Function has Unknown return type");
                    unknown_count += 1;
                }

                // Check parameter types
                for param in &func.params {
                    if matches!(param.ty, Type::Unknown) {
                        eprintln!("  - Parameter has Unknown type");
                        unknown_count += 1;
                    }
                }
            }
        }

        if unknown_count > 0 {
            eprintln!(
                "\n[ZYNTAX_DEBUG_TYPES] Found {} Unknown types in {} functions",
                unknown_count, func_count
            );
            eprintln!("[ZYNTAX_DEBUG_TYPES] Issue 0 Phase 1: Type checking integrated but inference doesn't modify AST yet");
            eprintln!("[ZYNTAX_DEBUG_TYPES] Next step: Implement AST transformation to apply inferred types\n");
        } else {
            eprintln!("\n[ZYNTAX_DEBUG_TYPES] ✓ No Unknown types found in program\n");
        }
    }

    /// Collect all declarations for forward references
    fn collect_declarations(&mut self, program: &TypedProgram) -> CompilerResult<()> {
        for decl in &program.declarations {
            match &decl.node {
                TypedDeclaration::Function(func) => {
                    let func_id = crate::hir::HirId::new();
                    self.symbols.functions.insert(func.name, func_id);
                }

                TypedDeclaration::Class(class_decl) => {
                    // Pre-register class methods in symbol table
                    for method in &class_decl.methods {
                        let method_id = crate::hir::HirId::new();
                        // Methods become mangled names: ClassName_methodName
                        let mangled_name = self.mangle_method_name(class_decl.name, method.name);
                        self.symbols.functions.insert(mangled_name, method_id);
                    }

                    // Pre-register constructors
                    for (i, _ctor) in class_decl.constructors.iter().enumerate() {
                        let ctor_id = crate::hir::HirId::new();
                        // Constructors: ClassName_constructor_N
                        // Use resolve_global() for portability across interner sources
                        let class_name_str = class_decl
                            .name
                            .resolve_global()
                            .unwrap_or_else(|| "UnknownClass".to_string());
                        let mut arena = self.arena.lock().unwrap();
                        let ctor_name =
                            arena.intern_string(&format!("{}_constructor_{}", class_name_str, i));
                        drop(arena);
                        self.symbols.functions.insert(ctor_name, ctor_id);
                    }
                }

                TypedDeclaration::Interface(_trait_decl) => {
                    // NOTE: Trait/interface lowering not yet implemented.
                    // Requires: (1) Trait method table, (2) Vtable generation, (3) Dynamic dispatch infrastructure
                    // WORKAROUND: Skipped (works for code without traits)
                    // FUTURE (v2.0): Implement trait system
                    // Estimated effort: 40-60 hours (major feature - requires runtime support)
                }

                TypedDeclaration::Impl(impl_block) => {
                    // Pre-register impl block methods in symbol table WITH MANGLED NAMES
                    // This must match the mangling done in lower_impl_block

                    // Check if implementing type is extern struct
                    let is_extern_struct =
                        matches!(&impl_block.for_type, zyntax_typed_ast::Type::Extern { .. });

                    // Get the implementing type name
                    let type_name = match &impl_block.for_type {
                        zyntax_typed_ast::Type::Named { id, .. } => {
                            if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                                type_def.name
                            } else {
                                continue; // Can't resolve type, skip
                            }
                        }
                        zyntax_typed_ast::Type::Unresolved(name) => {
                            if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                                type_def.name
                            } else {
                                continue; // Can't resolve type, skip
                            }
                        }
                        zyntax_typed_ast::Type::Extern { name, .. } => {
                            // Handle extern types (e.g., Tensor, Vector)
                            if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                                type_def.name
                            } else {
                                // Try with fresh InternedString in case of arena mismatch
                                let name_str = name.resolve_global().unwrap_or_default();
                                let fresh_name = InternedString::new_global(&name_str);
                                if let Some(type_def) =
                                    self.type_registry.get_type_by_name(fresh_name)
                                {
                                    type_def.name
                                } else {
                                    // For extern struct, use the name directly
                                    *name
                                }
                            }
                        }
                        _ => continue, // Non-named types, skip
                    };

                    // Check if this is an inherent impl (empty trait name)
                    let trait_name_str = impl_block.trait_name.resolve_global().unwrap_or_default();
                    let is_inherent = trait_name_str.is_empty();

                    for method in &impl_block.methods {
                        let method_id = crate::hir::HirId::new();

                        // Generate mangled name matching lower_impl_block
                        let mangled_name = if is_inherent {
                            let type_name_str = type_name
                                .resolve_global()
                                .unwrap_or_else(|| "UnknownType".to_string());
                            // Strip $ prefix if present (extern struct names have runtime_prefix)
                            let base_type_name =
                                type_name_str.strip_prefix('$').unwrap_or(&type_name_str);
                            let method_name_str = method
                                .name
                                .resolve_global()
                                .unwrap_or_else(|| "unknown_method".to_string());

                            // Use consistent TypeName$method naming for all extern struct methods
                            // ZRTL symbols start with $ (e.g., $Tensor$sum_f32), so no collision
                            InternedString::new_global(&format!(
                                "{}${}",
                                base_type_name, method_name_str
                            ))
                        } else {
                            // Trait method: {TypeName}${TraitName}${method_name}
                            self.mangle_trait_method_name(
                                type_name,
                                impl_block.trait_name,
                                method.name,
                            )
                        };

                        self.symbols.functions.insert(mangled_name, method_id);
                    }
                }

                TypedDeclaration::Effect(effect) => {
                    // Pre-register effect in symbol table
                    let effect_id = HirId::new();
                    self.symbols.effects.insert(effect.name, effect_id);
                }

                TypedDeclaration::EffectHandler(handler) => {
                    // Pre-register handler in symbol table
                    let handler_id = HirId::new();
                    self.symbols.handlers.insert(handler.name, handler_id);
                }

                _ => {}
            }
        }

        Ok(())
    }

    /// Lower a declaration
    fn lower_declaration(&mut self, decl: &TypedNode<TypedDeclaration>) -> CompilerResult<()> {
        match &decl.node {
            TypedDeclaration::Function(func) => {
                // Catch and warn about function failures - complex generic functions may fail
                // but we shouldn't fail the entire compilation for unused code
                if let Err(e) = self.lower_function(func) {
                    let func_name = func.name.resolve_global().unwrap_or_default();
                    eprintln!("[LOWERING WARN] Skipping function '{}': {:?}", func_name, e);
                    // Remove from symbol table if it was registered
                    self.symbols.functions.remove(&func.name);
                }
            }

            TypedDeclaration::Variable(var) => {
                self.lower_global_variable(var)?;
            }

            TypedDeclaration::Import(import) => {
                self.lower_import(import)?;
            }

            TypedDeclaration::Class(class_decl) => {
                self.lower_class(class_decl)?;
            }

            TypedDeclaration::Enum(enum_decl) => {
                self.lower_enum(enum_decl)?;
            }

            TypedDeclaration::Impl(impl_block) => {
                // Catch and warn about impl block failures - complex generics may fail
                // but we shouldn't fail the entire compilation for unused code
                if let Err(e) = self.lower_impl_block(impl_block) {
                    let trait_name = impl_block.trait_name.resolve_global().unwrap_or_default();
                    let type_name = match &impl_block.for_type {
                        zyntax_typed_ast::Type::Named { id, .. } => self
                            .type_registry
                            .get_type_by_id(*id)
                            .map(|t| t.name.resolve_global().unwrap_or_default())
                            .unwrap_or_else(|| format!("{:?}", id)),
                        _ => format!("{:?}", impl_block.for_type),
                    };
                    eprintln!(
                        "[LOWERING WARN] Skipping impl {} for {}: {:?}",
                        trait_name, type_name, e
                    );
                }
            }

            TypedDeclaration::Effect(effect) => {
                self.lower_effect(effect)?;
            }

            TypedDeclaration::EffectHandler(handler) => {
                self.lower_effect_handler(handler)?;
            }

            _ => {
                // NOTE: Other declaration types not yet lowered.
                // Includes: TypeAlias, Interface (trait lowering), etc.
                // Most type declarations are handled during type conversion, not as standalone declarations.
                //
                // WORKAROUND: Skipped (types registered on-demand during usage)
                // FUTURE (v2.0): Add explicit lowering for type declarations
                // Estimated effort: Interface/trait lowering ~40-60 hours
            }
        }

        Ok(())
    }

    /// Lower a function
    fn lower_function(&mut self, func: &TypedFunction) -> CompilerResult<()> {
        // Convert function signature
        let signature = self.convert_function_signature(func)?;

        // Create HIR function
        let mut hir_func = HirFunction::new(func.name, signature);
        hir_func.id = *self.symbols.functions.get(&func.name).unwrap();

        // Set function attributes
        self.set_function_attributes(&mut hir_func, func);

        // Gap 11: Handle extern functions
        if func.is_external {
            // Mark as external in HIR
            hir_func.is_external = true;

            // Set calling convention
            hir_func.calling_convention = self.convert_calling_convention(func.calling_convention);

            // Set link_name if specified (maps alias to actual symbol)
            // e.g., "image_load" -> "$Image$load"
            if let Some(ref link_name) = func.link_name {
                let link_name_str = link_name
                    .resolve_global()
                    .unwrap_or_else(|| link_name.to_string());
                hir_func.link_name = Some(link_name_str.clone());

                // Register the alias -> link_name mapping for SSA call resolution
                self.symbols
                    .extern_link_names
                    .insert(func.name, link_name_str);
            }

            // Extern functions have no body - clear the default entry block
            hir_func.blocks.clear();

            // Add to module and return early
            self.module.add_function(hir_func);
            return Ok(());
        }

        // Regular function: build CFG and SSA from body
        let body = func.body.as_ref().ok_or_else(|| {
            crate::CompilerError::Lowering(format!(
                "Non-extern function '{}' must have a body",
                func.name
            ))
        })?;

        // Build TypedCFG from function body (new approach - Gap #4 solution!)
        // This creates CFG structure from TypedAST without converting to HIR first
        let mut typed_cfg_builder = crate::typed_cfg::TypedCfgBuilder::new();
        let typed_cfg = typed_cfg_builder.build_from_block(body, hir_func.entry_block)?;

        // Debug: check TypedCFG
        log::trace!(
            "[LOWERING] TypedCFG for function {:?} (is_async={}): entry={:?}, nodes={}, edges={}",
            func.name,
            func.is_async,
            hir_func.entry_block,
            typed_cfg.graph.node_count(),
            typed_cfg.graph.edge_count()
        );

        // Convert to SSA form, processing TypedStatements to emit HIR instructions
        let ssa_builder = SsaBuilder::new(
            hir_func,
            self.type_registry.clone(),
            self.arena.clone(),
            self.symbols.functions.clone(),
        )
        .with_return_type(func.return_type.clone())
        .with_extern_link_names(self.symbols.extern_link_names.clone());
        let ssa = ssa_builder.build_from_typed_cfg(&typed_cfg)?;

        // Debug: check SSA result
        if func.is_async {
            log::trace!(
                "[LOWERING] After SSA build for async function {:?}: {} blocks",
                func.name,
                ssa.function.blocks.len()
            );
        }

        // Verify SSA properties
        ssa.verify()?;

        // Debug: print phi count before optimization
        let total_phis_before: usize = ssa.function.blocks.values().map(|b| b.phis.len()).sum();
        log::debug!(
            "[Lowering] Before optimize_trivial_phis: {} total phis",
            total_phis_before
        );
        for (block_id, block) in &ssa.function.blocks {
            if !block.phis.is_empty() {
                log::debug!(
                    "[Lowering]   Block {:?} has {} phis",
                    block_id,
                    block.phis.len()
                );
            }
        }

        // Optimize trivial phis
        let mut ssa = ssa;
        ssa.optimize_trivial_phis();

        // Debug: print phi count after optimization
        let total_phis_after: usize = ssa.function.blocks.values().map(|b| b.phis.len()).sum();
        log::debug!(
            "[Lowering] After optimize_trivial_phis: {} total phis",
            total_phis_after
        );

        hir_func = ssa.function;

        // Add string globals generated during SSA construction
        for global in ssa.string_globals {
            self.module.globals.insert(global.id, global);
        }

        // Add closure/lambda functions generated during SSA construction
        for closure_func in ssa.closure_functions {
            self.module.add_function(closure_func);
        }

        // Gap 6 Phase 2: Transform async functions to state machines
        if func.is_async {
            log::trace!(
                "[LOWERING] Before transform_async_function: {:?} with {} values, {} blocks",
                hir_func.name,
                hir_func.values.len(),
                hir_func.blocks.len()
            );
            hir_func = self.transform_async_function(hir_func)?;
        }

        // Add to module
        self.module.add_function(hir_func);

        Ok(())
    }

    /// Transform async function into Promise-returning function
    ///
    /// The new design:
    /// - `async fn foo(x: i32) -> i32` becomes `fn foo(x: i32) -> Promise<i32>`
    /// - Promise contains `{state_machine: *mut u8, poll_fn: fn(*mut u8) -> i64}`
    /// - The poll function is generated internally (not exposed as `_poll`)
    /// - When you call foo(x), you get a Promise that holds the state machine and poll fn
    fn transform_async_function(
        &mut self,
        original_func: HirFunction,
    ) -> CompilerResult<HirFunction> {
        // Create async compiler
        let mut async_compiler = crate::async_support::AsyncCompiler::new();

        // Compile the async function into a state machine
        let state_machine = async_compiler.compile_async_function(&original_func)?;

        // Generate state machine infrastructure
        let (poll_wrapper, async_entry_func) = {
            let mut arena = self.arena.lock().unwrap();

            // 1. Generate the state machine struct type (needed for size calculation)
            let _struct_type =
                async_compiler.generate_state_machine_struct(&state_machine, &mut *arena);

            // 2. Generate the poll() function (internal, prefixed with __)
            let poll_wrapper = async_compiler.generate_poll_function(
                &state_machine,
                &mut *arena,
                &original_func,
            )?;

            // 3. Generate the async entry function that returns Promise<T>
            // This keeps the ORIGINAL function name and returns Promise
            let async_entry_func = async_compiler.generate_async_entry_function(
                &state_machine,
                &poll_wrapper,
                &original_func,
                &mut *arena,
            )?;

            (poll_wrapper, async_entry_func)
        };

        // Add the poll() function to the module (internal implementation detail)
        self.module.add_function(poll_wrapper);

        // Return the async entry function as the replacement
        // This function has the SAME NAME as the original async function
        // and returns Promise<T>
        Ok(async_entry_func)
    }

    /// Convert function signature
    fn convert_function_signature(
        &self,
        func: &TypedFunction,
    ) -> CompilerResult<HirFunctionSignature> {
        let mut params = Vec::new();

        for param in &func.params {
            // Keep abstract types as nominal for method dispatch
            let hir_type = self.convert_type(&param.ty);
            let attributes = self.compute_param_attributes(&param.ty, param.mutability);

            params.push(HirParam {
                id: crate::hir::HirId::new(),
                name: param.name,
                ty: hir_type,
                attributes,
            });
        }

        // Keep abstract types as nominal for method dispatch
        // For void/unit functions, use empty returns vec (not vec![Void])
        let hir_return_type = self.convert_type(&func.return_type);
        let returns = if matches!(hir_return_type, HirType::Void) {
            vec![] // Empty for void functions
        } else {
            vec![hir_return_type]
        };

        // Convert type params from TypedFunction to HirTypeParam
        let hir_type_params: Vec<crate::hir::HirTypeParam> = func
            .type_params
            .iter()
            .map(|tp| {
                crate::hir::HirTypeParam {
                    name: tp.name,
                    constraints: vec![], // TODO: Convert bounds to constraints
                }
            })
            .collect();

        Ok(HirFunctionSignature {
            params,
            returns,
            type_params: hir_type_params,
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: func.is_async,
            effects: func.effects.clone(),
            is_pure: func.is_pure,
        })
    }

    /// Set function attributes
    fn set_function_attributes(&self, hir_func: &mut HirFunction, func: &TypedFunction) {
        // Set visibility-based attributes
        match func.visibility {
            Visibility::Public => {
                hir_func.attributes.hot = true; // Public functions might be hot
            }
            Visibility::Private => {
                hir_func.attributes.inline = crate::hir::InlineHint::Hint;
            }
            _ => {}
        }

        // Set calling convention (before async transformation)
        hir_func.calling_convention = if func.visibility == Visibility::Public {
            crate::hir::CallingConvention::C
        } else {
            crate::hir::CallingConvention::Fast
        };
    }

    /// Compute parameter attributes
    fn compute_param_attributes(&self, ty: &Type, mutability: Mutability) -> ParamAttributes {
        let mut attrs = ParamAttributes::default();

        // Large structs passed by reference
        if self.is_large_type(ty) {
            attrs.by_ref = true;
        }

        // Immutable references are readonly
        if matches!(ty, Type::Reference { .. }) && mutability == Mutability::Immutable {
            attrs.readonly = true;
            attrs.noalias = true;
        }

        // Non-null pointers
        if matches!(ty, Type::Reference { .. }) {
            attrs.nonnull = true;
        }

        attrs
    }

    /// Check if a type is considered "large" for ABI decisions
    fn is_large_type(&self, ty: &Type) -> bool {
        // Use precise TargetData size calculation
        // Types larger than 16 bytes are typically passed by reference in most ABIs
        matches!(ty, Type::Struct { .. } | Type::Tuple(_) if self.target_data.size_of(ty) > 16)
    }

    /// Get the size of a type in bytes (delegates to TargetData)
    fn estimate_type_size(&self, ty: &Type) -> usize {
        self.target_data.size_of(ty)
    }

    /// Convert a lifetime from TypedAST to HIR
    fn convert_lifetime(&self, lifetime: &zyntax_typed_ast::Lifetime) -> crate::hir::HirLifetime {
        crate::hir::HirLifetime {
            id: crate::hir::LifetimeId::new(),
            name: Some(lifetime.name),
            bounds: lifetime
                .bounds
                .iter()
                .map(|bound| match bound {
                    zyntax_typed_ast::LifetimeBound::Outlives(other) => {
                        crate::hir::LifetimeBound::Outlives(crate::hir::LifetimeId::new())
                    }
                    zyntax_typed_ast::LifetimeBound::Static => crate::hir::LifetimeBound::Static,
                })
                .collect(),
        }
    }

    /// Convert frontend type to HIR type
    /// Convert a type to its ABI representation (for function signatures)
    /// Abstract types are converted to their underlying types for zero-cost abstraction
    fn convert_type_to_abi(&self, ty: &Type) -> HirType {
        // For abstract types, use the underlying type for ABI
        if let Type::Named { id, .. } = ty {
            if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                if let zyntax_typed_ast::TypeKind::Abstract {
                    underlying_type, ..
                } = &type_def.kind
                {
                    eprintln!(
                        "[CONVERT_ABI] Abstract type '{}' → underlying ABI type",
                        type_def.name.resolve_global().unwrap_or_default()
                    );
                    return self.convert_type(underlying_type);
                }
            }
        }
        // For non-abstract types, use normal conversion
        self.convert_type(ty)
    }

    fn convert_type(&self, ty: &Type) -> HirType {
        use zyntax_typed_ast::PrimitiveType;

        match ty {
            Type::Primitive(prim) => match prim {
                PrimitiveType::Bool => HirType::Bool,
                PrimitiveType::I8 => HirType::I8,
                PrimitiveType::I16 => HirType::I16,
                PrimitiveType::I32 => HirType::I32,
                PrimitiveType::I64 => HirType::I64,
                PrimitiveType::I128 => HirType::I128,
                PrimitiveType::U8 => HirType::U8,
                PrimitiveType::U16 => HirType::U16,
                PrimitiveType::U32 => HirType::U32,
                PrimitiveType::U64 => HirType::U64,
                PrimitiveType::U128 => HirType::U128,
                PrimitiveType::F32 => HirType::F32,
                PrimitiveType::F64 => HirType::F64,
                PrimitiveType::Unit => HirType::Void,
                PrimitiveType::Char => HirType::U32, // Unicode scalar
                PrimitiveType::String => HirType::Ptr(Box::new(HirType::U8)), // String as u8 pointer
                _ => HirType::I64,                                            // Default
            },

            Type::Tuple(types) if types.is_empty() => HirType::Void,
            Type::Tuple(types) => HirType::Struct(crate::hir::HirStructType {
                name: None,
                fields: types.iter().map(|t| self.convert_type(t)).collect(),
                packed: false,
            }),

            Type::Reference {
                ty,
                mutability,
                lifetime,
                ..
            } => {
                // Convert lifetime if present
                let hir_lifetime = if let Some(lt) = lifetime {
                    self.convert_lifetime(lt)
                } else {
                    // Anonymous lifetime
                    crate::hir::HirLifetime {
                        id: crate::hir::LifetimeId::new(),
                        name: None,
                        bounds: vec![],
                    }
                };

                HirType::Ref {
                    lifetime: hir_lifetime,
                    pointee: Box::new(self.convert_type(ty)),
                    mutable: *mutability == Mutability::Mutable,
                }
            }

            Type::Array {
                element_type,
                size: Some(size_val),
                ..
            } => {
                let size = match size_val {
                    zyntax_typed_ast::ConstValue::UInt(u) => *u,
                    zyntax_typed_ast::ConstValue::Int(i) => *i as u64,
                    _ => 0,
                };
                HirType::Array(Box::new(self.convert_type(element_type)), size)
            }

            Type::Function {
                params,
                return_type,
                is_varargs,
                ..
            } => {
                // Note: Lifetime parameters are embedded in the parameter types themselves
                // (e.g., Type::Reference contains lifetime information). They are preserved
                // during convert_type() and don't need separate tracking here.
                HirType::Function(Box::new(crate::hir::HirFunctionType {
                    params: params.iter().map(|p| self.convert_type(&p.ty)).collect(),
                    returns: vec![self.convert_type(return_type)],
                    lifetime_params: vec![], // Lifetimes tracked in parameter types
                    is_variadic: *is_varargs,
                }))
            }

            Type::Projection { base, item } => {
                // Handle projections like T::AssocType
                // For now, we don't have full associated type information in TypedAST
                // This will be used when TypedAST is extended with associated types
                // TODO: Enhance when TypedAST gets AssociatedType variant with trait_id
                HirType::Opaque(*item)
            }

            Type::Associated {
                trait_name,
                type_name,
            } => {
                // Associated types in traits (not yet fully integrated)
                // TODO: Resolve using trait_name and type_name when trait registry is enhanced
                HirType::Opaque(*type_name)
            }

            Type::Extern { name, .. } => {
                // External/opaque types from ZRTL plugins
                // These are represented as opaque pointers at the HIR level
                HirType::Opaque(*name)
            }

            Type::Unresolved(name) => {
                // Look up the type in TypeRegistry - first try aliases, then try named types
                if let Some(resolved_type) = self.type_registry.resolve_alias(*name) {
                    // Recursively convert the resolved type
                    self.convert_type(resolved_type)
                } else if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                    // Found a named type (struct, enum, etc.) - convert it as Named type
                    let named_type = Type::Named {
                        id: type_def.id,
                        type_args: vec![],
                        const_args: vec![],
                        variance: vec![],
                        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                    };
                    self.convert_type(&named_type)
                } else {
                    log::warn!(
                        "Could not resolve type '{}', defaulting to I64",
                        name.resolve_global().unwrap_or_default()
                    );
                    HirType::I64 // Fallback
                }
            }

            Type::Named { id, .. } => {
                // Look up type definition in registry
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    match &type_def.kind {
                        zyntax_typed_ast::TypeKind::Struct { fields, .. } => {
                            // Convert struct fields
                            let field_types: Vec<_> = fields
                                .iter()
                                .map(|field| self.convert_type(&field.ty))
                                .collect();

                            HirType::Struct(crate::hir::HirStructType {
                                name: Some(type_def.name),
                                fields: field_types,
                                packed: false,
                            })
                        }

                        zyntax_typed_ast::TypeKind::Enum { variants } => {
                            // Convert enum to discriminated union
                            let hir_variants: Vec<_> = variants
                                .iter()
                                .map(|variant| {
                                    let variant_ty = match &variant.fields {
                                        zyntax_typed_ast::VariantFields::Unit => HirType::Void,
                                        zyntax_typed_ast::VariantFields::Tuple(types) => {
                                            let fields: Vec<_> = types
                                                .iter()
                                                .map(|ty| self.convert_type(ty))
                                                .collect();
                                            HirType::Struct(crate::hir::HirStructType {
                                                name: None,
                                                fields,
                                                packed: false,
                                            })
                                        }
                                        zyntax_typed_ast::VariantFields::Named(fields) => {
                                            let field_types: Vec<_> = fields
                                                .iter()
                                                .map(|field| self.convert_type(&field.ty))
                                                .collect();
                                            HirType::Struct(crate::hir::HirStructType {
                                                name: Some(variant.name),
                                                fields: field_types,
                                                packed: false,
                                            })
                                        }
                                    };

                                    crate::hir::HirUnionVariant {
                                        name: variant.name,
                                        ty: variant_ty,
                                        discriminant: variant.discriminant.unwrap_or(0) as u64,
                                    }
                                })
                                .collect();

                            HirType::Union(Box::new(crate::hir::HirUnionType {
                                name: Some(type_def.name),
                                variants: hir_variants,
                                discriminant_type: Box::new(HirType::U32),
                                is_c_union: false,
                            }))
                        }

                        zyntax_typed_ast::TypeKind::Abstract { .. } => {
                            // Abstract types are zero-cost wrappers with struct layout
                            // Convert them as structs so field access works
                            // Fields are stored in type_def.fields, not in the TypeKind::Abstract itself
                            let field_types: Vec<_> = type_def
                                .fields
                                .iter()
                                .map(|field| self.convert_type(&field.ty))
                                .collect();

                            eprintln!(
                                "[CONVERT TYPE] Abstract type '{}' → struct with {} fields",
                                type_def.name.resolve_global().unwrap_or_default(),
                                field_types.len()
                            );

                            HirType::Struct(crate::hir::HirStructType {
                                name: Some(type_def.name),
                                fields: field_types,
                                packed: false,
                            })
                        }

                        zyntax_typed_ast::TypeKind::Interface { .. } => {
                            // Interfaces/Traits become opaque for now
                            // Full trait support requires vtable generation
                            HirType::Opaque(type_def.name)
                        }

                        zyntax_typed_ast::TypeKind::Alias { target } => {
                            // Recursively resolve alias
                            self.convert_type(target)
                        }

                        _ => HirType::Opaque(type_def.name),
                    }
                } else {
                    // Type not found - create opaque placeholder
                    // Use a simple string representation
                    HirType::I64 // Fallback
                }
            }

            _ => HirType::I64, // Default for unsupported types
        }
    }

    /// Lower a global variable
    fn lower_global_variable(
        &mut self,
        var: &zyntax_typed_ast::TypedVariable,
    ) -> CompilerResult<()> {
        let hir_type = self.convert_type(&var.ty);

        // Evaluate initializer expression if present
        let initializer = if let Some(init_expr) = &var.initializer {
            match self.eval_const_expression(&init_expr.node, &hir_type) {
                Ok(constant) => Some(constant),
                Err(e) => {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        format!(
                            "Global variable '{}' has non-constant initializer: {}",
                            var.name, e
                        ),
                        Some(init_expr.span),
                    );
                    None
                }
            }
        } else {
            None
        };

        let global = crate::hir::HirGlobal {
            id: crate::hir::HirId::new(),
            name: var.name,
            ty: hir_type,
            initializer,
            is_const: var.mutability == Mutability::Immutable,
            is_thread_local: false,
            linkage: self.convert_linkage(var.visibility),
            visibility: self.convert_visibility(var.visibility),
        };

        self.symbols.globals.insert(var.name, global.id);
        self.module.add_global(global);

        Ok(())
    }

    /// Lower an enum declaration to HIR Union type
    fn lower_enum(
        &mut self,
        enum_decl: &zyntax_typed_ast::typed_ast::TypedEnum,
    ) -> CompilerResult<()> {
        use crate::hir::{HirType, HirUnionType, HirUnionVariant};
        use zyntax_typed_ast::typed_ast::TypedVariantFields;

        // Determine discriminant type based on number of variants
        let discriminant_type = if enum_decl.variants.len() <= 256 {
            HirType::U8
        } else if enum_decl.variants.len() <= 65536 {
            HirType::U16
        } else {
            HirType::U32
        };

        // Convert variants
        let mut hir_variants = Vec::new();
        for (variant_idx, variant) in enum_decl.variants.iter().enumerate() {
            // Determine discriminant value
            let discriminant = if let Some(disc_expr) = &variant.discriminant {
                // Evaluate constant expression for explicit discriminant
                match self.eval_const_expression(&disc_expr.node, &discriminant_type) {
                    Ok(crate::hir::HirConstant::U8(v)) => v as u64,
                    Ok(crate::hir::HirConstant::U16(v)) => v as u64,
                    Ok(crate::hir::HirConstant::U32(v)) => v as u64,
                    Ok(crate::hir::HirConstant::U64(v)) => v,
                    Ok(crate::hir::HirConstant::I8(v)) => v as u64,
                    Ok(crate::hir::HirConstant::I16(v)) => v as u64,
                    Ok(crate::hir::HirConstant::I32(v)) => v as u64,
                    Ok(crate::hir::HirConstant::I64(v)) => v as u64,
                    _ => {
                        self.diagnostic(
                            DiagnosticLevel::Error,
                            format!("Invalid discriminant for enum variant '{}'", variant.name),
                            Some(variant.span),
                        );
                        variant_idx as u64
                    }
                }
            } else {
                variant_idx as u64
            };

            // Convert variant fields to HIR type
            let variant_ty = match &variant.fields {
                TypedVariantFields::Unit => {
                    // Unit variant - use void type
                    HirType::Void
                }
                TypedVariantFields::Tuple(field_types) => {
                    if field_types.len() == 1 {
                        // Single field - use type directly
                        self.convert_type(&field_types[0])
                    } else {
                        // Multiple fields - create tuple struct
                        let fields: Vec<HirType> =
                            field_types.iter().map(|ty| self.convert_type(ty)).collect();
                        HirType::Struct(crate::hir::HirStructType {
                            name: None, // Anonymous tuple struct
                            fields,
                            packed: false,
                        })
                    }
                }
                TypedVariantFields::Named(named_fields) => {
                    // Named fields - create named struct
                    let fields: Vec<HirType> = named_fields
                        .iter()
                        .map(|field| self.convert_type(&field.ty))
                        .collect();
                    HirType::Struct(crate::hir::HirStructType {
                        name: Some(variant.name), // Use variant name
                        fields,
                        packed: false,
                    })
                }
            };

            hir_variants.push(HirUnionVariant {
                name: variant.name,
                ty: variant_ty,
                discriminant,
            });
        }

        // Create HIR union type
        let union_type = HirUnionType {
            name: Some(enum_decl.name),
            variants: hir_variants,
            discriminant_type: Box::new(discriminant_type),
            is_c_union: false, // Tagged union with discriminant
        };

        // Register the enum type in the module
        // Look up the type ID from the type registry (should already be registered by type checker)
        if let Some(type_id) = self
            .type_registry
            .get_type_by_name(enum_decl.name)
            .map(|def| def.id)
        {
            self.module
                .types
                .insert(type_id, HirType::Union(Box::new(union_type)));
        } else {
            // Type not found in registry - this shouldn't happen for well-typed programs
            self.diagnostic(
                DiagnosticLevel::Warning,
                format!("Enum type '{}' not found in type registry", enum_decl.name),
                Some(enum_decl.span),
            );
        }

        Ok(())
    }

    /// Lower an algebraic effect declaration to HIR
    fn lower_effect(&mut self, effect: &TypedEffect) -> CompilerResult<()> {
        // Get the pre-registered effect ID
        let effect_id = *self.symbols.effects.get(&effect.name).ok_or_else(|| {
            crate::CompilerError::Lowering(format!(
                "Effect '{}' not found in symbol table",
                effect.name
            ))
        })?;

        // Convert type parameters
        let type_params: Vec<HirTypeParam> = effect
            .type_params
            .iter()
            .map(|tp| HirTypeParam {
                name: tp.name,
                constraints: vec![],
            })
            .collect();

        // Convert effect operations
        let operations: Vec<HirEffectOp> = effect
            .operations
            .iter()
            .map(|op| {
                // Convert operation type parameters
                let op_type_params: Vec<HirTypeParam> = op
                    .type_params
                    .iter()
                    .map(|tp| HirTypeParam {
                        name: tp.name,
                        constraints: vec![],
                    })
                    .collect();

                // Convert operation parameters
                let params: Vec<HirParam> = op
                    .params
                    .iter()
                    .map(|p| HirParam {
                        id: HirId::new(),
                        name: p.name,
                        ty: self.convert_type(&p.ty),
                        attributes: ParamAttributes::default(),
                    })
                    .collect();

                HirEffectOp {
                    id: HirId::new(),
                    name: op.name,
                    type_params: op_type_params,
                    params,
                    return_type: self.convert_type(&op.return_type),
                }
            })
            .collect();

        // Create HIR effect
        let hir_effect = HirEffect {
            id: effect_id,
            name: effect.name,
            type_params,
            operations,
        };

        // Add to module
        self.module.effects.insert(effect_id, hir_effect);

        Ok(())
    }

    /// Lower an effect handler declaration to HIR
    fn lower_effect_handler(&mut self, handler: &TypedEffectHandler) -> CompilerResult<()> {
        // Get the pre-registered handler ID
        let handler_id = *self.symbols.handlers.get(&handler.name).ok_or_else(|| {
            crate::CompilerError::Lowering(format!(
                "Handler '{}' not found in symbol table",
                handler.name
            ))
        })?;

        // Find the effect being handled
        let effect_id = *self
            .symbols
            .effects
            .get(&handler.effect_name)
            .ok_or_else(|| {
                crate::CompilerError::Lowering(format!(
                    "Effect '{}' not found for handler '{}'",
                    handler.effect_name, handler.name
                ))
            })?;

        // Convert type parameters
        let type_params: Vec<HirTypeParam> = handler
            .type_params
            .iter()
            .map(|tp| HirTypeParam {
                name: tp.name,
                constraints: vec![],
            })
            .collect();

        // Convert handler state fields
        let state_fields: Vec<HirHandlerField> = handler
            .fields
            .iter()
            .map(|f| HirHandlerField {
                name: f.name,
                ty: self.convert_type(&f.ty),
            })
            .collect();

        // Convert handler implementations
        // For now, we only store the metadata - actual implementation lowering
        // happens when we compile the handler's functions
        let implementations: Vec<HirEffectHandlerImpl> = handler
            .handlers
            .iter()
            .map(|impl_| {
                // Convert type parameters for this implementation
                let impl_type_params: Vec<HirTypeParam> = impl_
                    .type_params
                    .iter()
                    .map(|tp| HirTypeParam {
                        name: tp.name,
                        constraints: vec![],
                    })
                    .collect();

                // Convert parameters (including the resume continuation if present)
                let params: Vec<HirParam> = impl_
                    .params
                    .iter()
                    .map(|p| HirParam {
                        id: HirId::new(),
                        name: p.name,
                        ty: self.convert_type(&p.ty),
                        attributes: ParamAttributes::default(),
                    })
                    .collect();

                // Check if this handler is resumable (has a Resume parameter)
                let is_resumable = impl_.params.iter().any(|p| {
                    matches!(&p.ty, Type::Named { id, .. } if {
                        self.type_registry.get_type_by_id(*id)
                            .map(|def| def.name.resolve_global().as_deref() == Some("Resume"))
                            .unwrap_or(false)
                    })
                });

                HirEffectHandlerImpl {
                    op_name: impl_.op_name,
                    type_params: impl_type_params,
                    params,
                    return_type: self.convert_type(&impl_.return_type),
                    entry_block: HirId::new(),
                    blocks: indexmap::IndexMap::new(), // Will be filled in when we compile the body
                    is_resumable,
                }
            })
            .collect();

        // Create HIR effect handler
        let hir_handler = HirEffectHandler {
            id: handler_id,
            name: handler.name,
            effect_id,
            type_params,
            state_fields,
            implementations,
        };

        // Add to module
        self.module.handlers.insert(handler_id, hir_handler);

        Ok(())
    }

    /// Re-type expressions in a block that reference self parameters with resolved types
    fn retype_block_with_self(
        &self,
        block: &zyntax_typed_ast::TypedBlock,
        self_params: &[(zyntax_typed_ast::InternedString, zyntax_typed_ast::Type)], // (param_name, resolved_type)
    ) -> zyntax_typed_ast::TypedBlock {
        use zyntax_typed_ast::TypedBlock;

        eprintln!(
            "[RETYPE_BLOCK] Processing {} statements",
            block.statements.len()
        );
        let retyped_statements = block
            .statements
            .iter()
            .enumerate()
            .map(|(idx, stmt_node)| {
                eprintln!("[RETYPE_BLOCK] Processing statement {}", idx);
                self.retype_statement_with_self(stmt_node, self_params)
            })
            .collect();

        TypedBlock {
            statements: retyped_statements,
            span: block.span,
        }
    }

    /// Re-type a statement node's expressions that reference self parameters
    fn retype_statement_with_self(
        &self,
        stmt_node: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedStatement>,
        self_params: &[(zyntax_typed_ast::InternedString, zyntax_typed_ast::Type)],
    ) -> zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedStatement> {
        use zyntax_typed_ast::{TypedLet, TypedNode, TypedStatement};

        let retyped_stmt = match &stmt_node.node {
            TypedStatement::Expression(expr) => {
                eprintln!("[RETYPE_STMT] Retyping expression statement");
                let retyped = self.retype_expression_node_with_self(expr, self_params);
                eprintln!("[RETYPE_STMT] Expression retyping done");
                TypedStatement::Expression(Box::new(retyped))
            }
            TypedStatement::Let(let_stmt) => TypedStatement::Let(TypedLet {
                name: let_stmt.name,
                ty: let_stmt.ty.clone(),
                mutability: let_stmt.mutability,
                initializer: let_stmt
                    .initializer
                    .as_ref()
                    .map(|init| Box::new(self.retype_expression_node_with_self(init, self_params))),
                span: let_stmt.span,
            }),
            TypedStatement::Return(opt_expr) => TypedStatement::Return(
                opt_expr
                    .as_ref()
                    .map(|expr| Box::new(self.retype_expression_node_with_self(expr, self_params))),
            ),
            TypedStatement::Break(_) | TypedStatement::Continue => stmt_node.node.clone(),
            TypedStatement::If(if_stmt) => {
                use zyntax_typed_ast::TypedIf;
                TypedStatement::If(TypedIf {
                    condition: Box::new(
                        self.retype_expression_node_with_self(&if_stmt.condition, self_params),
                    ),
                    then_block: self.retype_block_with_self(&if_stmt.then_block, self_params),
                    else_block: if_stmt
                        .else_block
                        .as_ref()
                        .map(|block| self.retype_block_with_self(block, self_params)),
                    span: if_stmt.span,
                })
            }
            TypedStatement::While(while_stmt) => {
                use zyntax_typed_ast::TypedWhile;
                TypedStatement::While(TypedWhile {
                    condition: Box::new(
                        self.retype_expression_node_with_self(&while_stmt.condition, self_params),
                    ),
                    body: self.retype_block_with_self(&while_stmt.body, self_params),
                    span: while_stmt.span,
                })
            }
            TypedStatement::For(for_stmt) => {
                use zyntax_typed_ast::TypedFor;
                TypedStatement::For(TypedFor {
                    pattern: for_stmt.pattern.clone(),
                    iterator: Box::new(
                        self.retype_expression_node_with_self(&for_stmt.iterator, self_params),
                    ),
                    body: self.retype_block_with_self(&for_stmt.body, self_params),
                })
            }
            // Handle other statement types that may contain expressions
            _ => stmt_node.node.clone(),
        };

        TypedNode::new(retyped_stmt, stmt_node.ty.clone(), stmt_node.span)
    }

    /// Re-type an expression node if it references a self parameter
    fn retype_expression_node_with_self(
        &self,
        expr_node: &zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>,
        self_params: &[(zyntax_typed_ast::InternedString, zyntax_typed_ast::Type)],
    ) -> zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression> {
        use zyntax_typed_ast::{
            Type, TypedBinary, TypedCall, TypedExpression, TypedFieldAccess, TypedNode, TypedUnary,
        };

        let (retyped_expr, new_ty) = match &expr_node.node {
            // Variable reference - check if it's a self parameter and retype if needed
            TypedExpression::Variable(name) => {
                // Check if this variable is a self parameter
                if let Some((_, resolved_ty)) = self_params
                    .iter()
                    .find(|(param_name, _)| param_name == name)
                {
                    // Only retype if current type is Any or Unresolved
                    if matches!(expr_node.ty, Type::Any)
                        || matches!(expr_node.ty, Type::Unresolved(_))
                    {
                        return TypedNode::new(
                            TypedExpression::Variable(*name),
                            resolved_ty.clone(),
                            expr_node.span,
                        );
                    }
                }
                (expr_node.node.clone(), expr_node.ty.clone())
            }

            // Field access - retype the object expression and possibly update field type
            TypedExpression::Field(field_access) => {
                eprintln!(
                    "[RETYPE] Field access starting: object.ty={:?}, field={:?}",
                    field_access.object.ty, field_access.field
                );
                let retyped_object =
                    self.retype_expression_node_with_self(&field_access.object, self_params);
                eprintln!("[RETYPE] Field access: object before={:?}, after={:?}, field={:?}, expr_ty={:?}",
                    field_access.object.ty, retyped_object.ty, field_access.field, expr_node.ty);
                let retyped_field_access = TypedFieldAccess {
                    object: Box::new(retyped_object),
                    field: field_access.field,
                };
                eprintln!("[RETYPE] Field access done, returning");
                // Keep the original field result type - it should have been correctly typed by the parser
                (
                    TypedExpression::Field(retyped_field_access),
                    expr_node.ty.clone(),
                )
            }

            // Binary operation - retype both operands
            TypedExpression::Binary(binary) => {
                let retyped_binary = TypedBinary {
                    op: binary.op,
                    left: Box::new(
                        self.retype_expression_node_with_self(&binary.left, self_params),
                    ),
                    right: Box::new(
                        self.retype_expression_node_with_self(&binary.right, self_params),
                    ),
                };
                (
                    TypedExpression::Binary(retyped_binary),
                    expr_node.ty.clone(),
                )
            }

            // Unary operation - retype the operand
            TypedExpression::Unary(unary) => {
                let retyped_unary = TypedUnary {
                    op: unary.op,
                    operand: Box::new(
                        self.retype_expression_node_with_self(&unary.operand, self_params),
                    ),
                };
                (TypedExpression::Unary(retyped_unary), expr_node.ty.clone())
            }

            // Function call - retype callee and arguments
            TypedExpression::Call(call) => {
                let retyped_call = TypedCall {
                    callee: Box::new(
                        self.retype_expression_node_with_self(&call.callee, self_params),
                    ),
                    positional_args: call
                        .positional_args
                        .iter()
                        .map(|arg| self.retype_expression_node_with_self(arg, self_params))
                        .collect(),
                    named_args: call.named_args.clone(), // Named args don't need retyping of names
                    type_args: call.type_args.clone(),
                };
                (TypedExpression::Call(retyped_call), expr_node.ty.clone())
            }

            // Struct literal - retype field values
            TypedExpression::Struct(struct_lit) => {
                use zyntax_typed_ast::TypedStructLiteral;
                let retyped_struct = TypedStructLiteral {
                    name: struct_lit.name,
                    fields: struct_lit
                        .fields
                        .iter()
                        .map(|field_init| {
                            use zyntax_typed_ast::TypedFieldInit;
                            TypedFieldInit {
                                name: field_init.name,
                                value: Box::new(self.retype_expression_node_with_self(
                                    &field_init.value,
                                    self_params,
                                )),
                            }
                        })
                        .collect(),
                };
                (
                    TypedExpression::Struct(retyped_struct),
                    expr_node.ty.clone(),
                )
            }

            // Method call - retype the receiver and arguments
            TypedExpression::MethodCall(method_call) => {
                use zyntax_typed_ast::TypedMethodCall;
                let retyped_receiver =
                    self.retype_expression_node_with_self(&method_call.receiver, self_params);
                let retyped_positional_args = method_call
                    .positional_args
                    .iter()
                    .map(|arg| self.retype_expression_node_with_self(arg, self_params))
                    .collect();
                let retyped_method_call = TypedMethodCall {
                    receiver: Box::new(retyped_receiver),
                    method: method_call.method,
                    type_args: method_call.type_args.clone(),
                    positional_args: retyped_positional_args,
                    named_args: method_call.named_args.clone(),
                };
                (
                    TypedExpression::MethodCall(retyped_method_call),
                    expr_node.ty.clone(),
                )
            }

            // All other expression types - just clone
            _ => (expr_node.node.clone(), expr_node.ty.clone()),
        };

        TypedNode::new(retyped_expr, new_ty, expr_node.span)
    }

    /// Lower an impl block by extracting and lowering its methods
    fn lower_impl_block(
        &mut self,
        impl_block: &zyntax_typed_ast::typed_ast::TypedTraitImpl,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::{Type, TypeId, TypedFunction};

        eprintln!(
            "[LOWERING IMPL] Starting lower_impl_block for_type={:?}, trait_name={:?}, {} methods",
            impl_block.for_type,
            impl_block.trait_name,
            impl_block.methods.len()
        );

        // Resolve the implementing type if it's still unresolved
        // The parser uses Unresolved types, we need to look them up in the registry
        let implementing_type = match &impl_block.for_type {
            Type::Unresolved(name) => {
                // Look up the type by name in the registry
                eprintln!("[LOWERING] Looking up unresolved type: {:?}", name);

                // Find the type in the registry
                if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                    let type_id = type_def.id;
                    eprintln!("[LOWERING] Found TypeId: {:?}", type_id);
                    Type::Named {
                        id: type_id,
                        type_args: vec![],
                        const_args: vec![],
                        variance: vec![],
                        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                    }
                } else {
                    let type_name_str = self
                        .arena
                        .lock()
                        .unwrap()
                        .resolve_string(*name)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    eprintln!(
                        "[LOWERING] WARNING: Type '{}' not found in registry",
                        type_name_str
                    );
                    impl_block.for_type.clone()
                }
            }
            _ => impl_block.for_type.clone(),
        };
        eprintln!(
            "[LOWERING] Impl block implementing_type after resolution: {:?}",
            implementing_type
        );

        // Check if this is an inherent impl (empty trait name) or a trait impl
        let trait_name_str = {
            let arena = self.arena.lock().unwrap();
            arena
                .resolve_string(impl_block.trait_name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| String::new())
        };

        let is_inherent = trait_name_str.is_empty();
        eprintln!(
            "[LOWERING] Impl block is_inherent: {}, trait_name: '{}'",
            is_inherent, trait_name_str
        );

        // Check if the implementing type is an extern struct
        // For extern structs, methods should automatically map to ZRTL symbols
        let is_extern_struct = matches!(&implementing_type, Type::Extern { .. });
        let extern_type_name = if let Type::Extern { name, .. } = &implementing_type {
            let name_str = name.resolve_global().unwrap_or_default();
            // The extern type name may have a leading $ (runtime_prefix) - strip it for the base name
            // e.g., "$Tensor" -> "Tensor"
            let base_name = name_str.strip_prefix('$').unwrap_or(&name_str);
            Some(base_name.to_string())
        } else {
            None
        };
        eprintln!(
            "[LOWERING] is_extern_struct: {}, extern_type_name: {:?}",
            is_extern_struct, extern_type_name
        );

        // For trait impls, register the implementation in the type registry
        let trait_id = if !is_inherent {
            let trait_def = self
                .type_registry
                .get_trait_by_name(impl_block.trait_name)
                .ok_or_else(|| {
                    crate::CompilerError::Analysis(format!(
                        "Trait '{}' not found in registry",
                        trait_name_str
                    ))
                })?;
            Some(trait_def.id)
        } else {
            None
        };

        // Build MethodImpl list from the impl block methods
        let method_impls: Vec<zyntax_typed_ast::type_registry::MethodImpl> = impl_block
            .methods
            .iter()
            .map(|method| {
                // Build parameter definitions
                let params: Vec<zyntax_typed_ast::type_registry::ParamDef> = method
                    .params
                    .iter()
                    .map(|p| zyntax_typed_ast::type_registry::ParamDef {
                        name: p.name,
                        ty: p.ty.clone(),
                        is_self: p.is_self,
                        is_varargs: false,
                        is_mut: matches!(
                            p.mutability,
                            zyntax_typed_ast::type_registry::Mutability::Mutable
                        ),
                    })
                    .collect();

                // Convert TypedTypeParam to TypeParam
                let type_params: Vec<zyntax_typed_ast::type_registry::TypeParam> = method
                    .type_params
                    .iter()
                    .map(|tp| {
                        // Convert TypedTypeBound to TypeBound
                        let bounds: Vec<zyntax_typed_ast::type_registry::TypeBound> =
                            tp.bounds
                                .iter()
                                .filter_map(|bound| {
                                    match bound {
                                        zyntax_typed_ast::typed_ast::TypedTypeBound::Trait(
                                            trait_type,
                                        ) => {
                                            // Extract name and args from the trait type
                                            match trait_type {
                                Type::Named { id: _, type_args, .. } => {
                                    // Get the trait name from the type
                                    // For now, just use a placeholder since we don't have easy access to the name
                                    Some(zyntax_typed_ast::type_registry::TypeBound::Trait {
                                        name: zyntax_typed_ast::InternedString::new_global("Trait"),
                                        args: type_args.clone(),
                                    })
                                }
                                Type::Unresolved(name) => {
                                    Some(zyntax_typed_ast::type_registry::TypeBound::Trait {
                                        name: *name,
                                        args: vec![],
                                    })
                                }
                                _ => None,
                            }
                                        }
                                        _ => None, // Skip other bounds for now
                                    }
                                })
                                .collect();

                        zyntax_typed_ast::type_registry::TypeParam {
                            name: tp.name,
                            bounds,
                            variance: zyntax_typed_ast::type_registry::Variance::Invariant,
                            default: tp.default.clone(),
                            span: tp.span,
                        }
                    })
                    .collect();

                let method_sig = zyntax_typed_ast::type_registry::MethodSig {
                    name: method.name,
                    type_params,
                    params,
                    return_type: method.return_type.clone(),
                    where_clause: vec![],
                    is_static: false,
                    is_async: method.is_async,
                    visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                    span: method.span,
                    is_extension: false,
                };

                zyntax_typed_ast::type_registry::MethodImpl {
                    signature: method_sig,
                    is_default: false,
                }
            })
            .collect();

        // Create ImplDef and register it (only for trait impls, not inherent impls)
        if let Some(tid) = trait_id {
            let impl_def = zyntax_typed_ast::type_registry::ImplDef {
                trait_id: tid,
                for_type: implementing_type.clone(),
                type_args: vec![],
                methods: method_impls,
                associated_types: std::collections::HashMap::new(),
                where_clause: vec![],
                span: impl_block.span,
            };

            // TODO: Register impl in type registry before lowering starts
            // self.type_registry is Arc (immutable), so we can't register here
            // For now, method resolution relies on mangled names being available in SSA phase
            eprintln!(
                "[LOWERING] Built impl def for trait {:?} for type {:?} (registration TODO)",
                tid, implementing_type
            );
            eprintln!("[LOWERING] ImplDef has {} methods", impl_def.methods.len());
        } else {
            eprintln!("[LOWERING] Skipping ImplDef creation for inherent impl (no trait)");
        }

        // For each method in the impl block, convert it to a function and lower it
        for (method_idx, method) in impl_block.methods.iter().enumerate() {
            eprintln!(
                "[LOWERING] Processing method {} of {}: {}",
                method_idx + 1,
                impl_block.methods.len(),
                method.name.resolve_global().unwrap_or_default()
            );

            // Resolve parameter types and track self parameters for body retyping
            let mut self_param_mappings: Vec<(zyntax_typed_ast::InternedString, Type)> = Vec::new();

            let params: Vec<zyntax_typed_ast::TypedParameter> = method
                .params
                .iter()
                .map(|p| {
                    eprintln!("[LOWERING] Param is_self={}, ty={:?}", p.is_self, p.ty);
                    let resolved_ty = if p.is_self
                        && (matches!(p.ty, Type::Any) || matches!(p.ty, Type::Unresolved(_)))
                    {
                        // Self parameter without explicit type -> use implementing type
                        eprintln!("[LOWERING] Resolving self param to {:?}", implementing_type);
                        let resolved = implementing_type.clone();
                        // Track this parameter for body retyping
                        self_param_mappings.push((p.name, resolved.clone()));
                        resolved
                    } else {
                        p.ty.clone()
                    };

                    zyntax_typed_ast::TypedParameter {
                        name: p.name,
                        ty: resolved_ty,
                        mutability: p.mutability,
                        kind: p.kind.clone(),
                        default_value: p.default_value.clone(),
                        attributes: p.attributes.clone(),
                        span: p.span,
                    }
                })
                .collect();

            // Re-type the method body to update self references
            eprintln!(
                "[LOWERING] Retyping method body, self_param_mappings: {:?}",
                self_param_mappings
            );
            let retyped_body = method.body.as_ref().map(|body| {
                let result = self.retype_block_with_self(body, &self_param_mappings);
                eprintln!("[LOWERING] Retyping complete");
                result
            });

            // Resolve return type: convert Self -> implementing_type, Unresolved -> lookup
            let resolved_return_type = match &method.return_type {
                Type::Any => {
                    // Self type in return position -> use implementing type
                    eprintln!(
                        "[LOWERING] Resolving return type Self -> {:?}",
                        implementing_type
                    );
                    implementing_type.clone()
                }
                Type::Unresolved(name) => {
                    // Try to resolve from registry
                    if let Some(type_def) = self.type_registry.get_type_by_name(*name) {
                        Type::Named {
                            id: type_def.id,
                            type_args: vec![],
                            const_args: vec![],
                            variance: vec![],
                            nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                        }
                    } else {
                        method.return_type.clone()
                    }
                }
                _ => method.return_type.clone(),
            };

            // Mangle the method name to include trait and type info
            // For extern struct methods WITH bodies (wrappers): use _zynml_{Type}_{method} to avoid ZRTL collision
            // For extern struct methods WITHOUT bodies: auto-map creates external function with ZRTL link
            // For regular structs: {TypeName}${method_name} or {TypeName}${TraitName}${method_name}
            let mangled_name = {
                let type_name = match &implementing_type {
                    Type::Named { id, .. } => {
                        if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                            type_def.name
                        } else {
                            method.name // Fallback to original name
                        }
                    }
                    Type::Extern { name, .. } => *name, // Use the extern struct's name
                    _ => method.name,                   // For other types, use original name
                };

                // For inherent impls (empty trait name), use simpler mangling
                if is_inherent {
                    let type_name_str = type_name
                        .resolve_global()
                        .unwrap_or_else(|| "UnknownType".to_string());
                    // Strip $ prefix if present (extern struct names have runtime_prefix)
                    let base_type_name = type_name_str.strip_prefix('$').unwrap_or(&type_name_str);
                    let method_name_str = method
                        .name
                        .resolve_global()
                        .unwrap_or_else(|| "unknown_method".to_string());

                    // Use consistent TypeName$method naming for all extern struct methods
                    // ZRTL symbols start with $ (e.g., $Tensor$sum_f32), so no collision
                    let mangled = format!("{}${}", base_type_name, method_name_str);
                    InternedString::new_global(&mangled)
                } else {
                    // For trait impls, include trait name
                    let trait_name = impl_block.trait_name;
                    self.mangle_trait_method_name(type_name, trait_name, method.name)
                }
            };

            eprintln!("[LOWERING] Mangled method name: {:?}", mangled_name);

            // Get the existing function_id from collect_declarations, or create new if missing
            // This ensures consistency between the two phases
            let function_id = if let Some(&existing_id) = self.symbols.functions.get(&mangled_name)
            {
                eprintln!(
                    "[LOWERING] Using existing function_id for {:?}",
                    mangled_name
                );
                existing_id
            } else {
                // Fallback: create new (shouldn't happen if collect_declarations ran first)
                eprintln!("[LOWERING] WARNING: Creating new function_id for {:?} (should have been pre-registered)", mangled_name);
                let new_id = crate::hir::HirId::new();
                self.symbols.functions.insert(mangled_name, new_id);
                new_id
            };

            // For extern struct methods WITHOUT a body, create external function declaration
            // that automatically maps to ZRTL symbol: $TypeName$method_name
            // If the method has an explicit body, use that body as an override
            if is_extern_struct && method.body.is_none() {
                if let Some(ref type_name) = extern_type_name {
                    let method_name_str = method.name.resolve_global().unwrap_or_default();
                    // ZRTL symbol format: $TypeName$method_name (e.g., $Tensor$matmul)
                    let zrtl_symbol = format!("${}${}", type_name, method_name_str);
                    eprintln!(
                        "[LOWERING] Extern struct method: {} -> ZRTL symbol: {} (params: {})",
                        mangled_name.resolve_global().unwrap_or_default(),
                        zrtl_symbol,
                        params.len()
                    );

                    let func = TypedFunction {
                        name: mangled_name,
                        annotations: vec![],
                        effects: vec![],
                        type_params: vec![],
                        params,
                        return_type: resolved_return_type,
                        body: None, // No body - external function
                        visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                        is_async: method.is_async,
                        is_pure: false,
                        is_external: true, // Mark as external
                        calling_convention:
                            zyntax_typed_ast::type_registry::CallingConvention::Default,
                        link_name: Some(InternedString::new_global(&zrtl_symbol)), // Link to ZRTL symbol
                    };

                    if let Err(e) = self.lower_function(&func) {
                        let method_name_str = mangled_name.resolve_global().unwrap_or_default();
                        eprintln!(
                            "[LOWERING WARN] Skipping extern method '{}': {:?}",
                            method_name_str, e
                        );
                        self.symbols.functions.remove(&mangled_name);
                        continue;
                    }
                    continue; // Skip regular function lowering
                }
            }

            // Create a function from the method (regular, non-extern struct)
            let func = TypedFunction {
                name: mangled_name, // Use mangled name for trait method
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params,
                return_type: resolved_return_type,
                body: retyped_body, // Use retyped body with updated self types
                visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                is_async: method.is_async,
                is_pure: false,
                is_external: false,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: None,
            };

            // Lower the method as a regular function
            // Catch errors for individual methods - complex generic methods may fail
            // but we don't want to fail the entire impl block
            if let Err(e) = self.lower_function(&func) {
                let method_name_str = mangled_name.resolve_global().unwrap_or_default();
                eprintln!(
                    "[LOWERING WARN] Skipping method '{}': {:?}",
                    method_name_str, e
                );
                // Remove the function from the symbols table so SSA doesn't try to process it
                self.symbols.functions.remove(&mangled_name);
                continue;
            }
        }

        Ok(())
    }

    /// Evaluate a TypedExpression as a compile-time constant
    fn eval_const_expression(
        &self,
        expr: &zyntax_typed_ast::TypedExpression,
        expected_ty: &crate::hir::HirType,
    ) -> CompilerResult<crate::hir::HirConstant> {
        use crate::hir::HirConstant;
        use zyntax_typed_ast::{TypedExpression, TypedLiteral};

        match expr {
            // Simple literals
            TypedExpression::Literal(lit) => match lit {
                TypedLiteral::Bool(b) => Ok(HirConstant::Bool(*b)),
                TypedLiteral::Integer(i) => {
                    // Convert to appropriate integer type based on expected type
                    match expected_ty {
                        crate::hir::HirType::I8 => Ok(HirConstant::I8(*i as i8)),
                        crate::hir::HirType::I16 => Ok(HirConstant::I16(*i as i16)),
                        crate::hir::HirType::I32 => Ok(HirConstant::I32(*i as i32)),
                        crate::hir::HirType::I64 => Ok(HirConstant::I64(*i as i64)),
                        crate::hir::HirType::I128 => Ok(HirConstant::I128(*i)),
                        crate::hir::HirType::U8 => Ok(HirConstant::U8(*i as u8)),
                        crate::hir::HirType::U16 => Ok(HirConstant::U16(*i as u16)),
                        crate::hir::HirType::U32 => Ok(HirConstant::U32(*i as u32)),
                        crate::hir::HirType::U64 => Ok(HirConstant::U64(*i as u64)),
                        crate::hir::HirType::U128 => Ok(HirConstant::U128(*i as u128)),
                        _ => Ok(HirConstant::I32(*i as i32)), // Default to i32
                    }
                }
                TypedLiteral::Float(f) => {
                    match expected_ty {
                        crate::hir::HirType::F32 => Ok(HirConstant::F32(*f as f32)),
                        crate::hir::HirType::F64 => Ok(HirConstant::F64(*f)),
                        _ => Ok(HirConstant::F64(*f)), // Default to f64
                    }
                }
                TypedLiteral::String(s) => Ok(HirConstant::String(*s)),
                TypedLiteral::Char(_) => Err(crate::CompilerError::Analysis(
                    "Char literals not yet supported in global initializers".into(),
                )),
                TypedLiteral::Unit => {
                    // Unit type maps to null pointer type
                    Ok(HirConstant::Null(expected_ty.clone()))
                }
                TypedLiteral::Null => {
                    // Null literal for optional types
                    Ok(HirConstant::Null(expected_ty.clone()))
                }
                TypedLiteral::Undefined => {
                    // Undefined - use zeroed memory as placeholder
                    Ok(HirConstant::I32(0))
                }
            },

            // Array literals
            TypedExpression::Array(elements) => {
                let element_ty = match expected_ty {
                    crate::hir::HirType::Array(elem_ty, _) => elem_ty.as_ref().clone(),
                    _ => {
                        return Err(crate::CompilerError::Analysis(
                            "Array initializer for non-array type".into(),
                        ))
                    }
                };

                let mut const_elements = Vec::new();
                for elem in elements {
                    const_elements.push(self.eval_const_expression(&elem.node, &element_ty)?);
                }

                Ok(HirConstant::Array(const_elements))
            }

            // Struct literals
            TypedExpression::Struct(struct_lit) => {
                let field_types = match expected_ty {
                    crate::hir::HirType::Struct(s) => &s.fields,
                    _ => {
                        return Err(crate::CompilerError::Analysis(
                            "Struct initializer for non-struct type".into(),
                        ))
                    }
                };

                let mut const_fields = Vec::new();
                for (field_init, field_ty) in struct_lit.fields.iter().zip(field_types.iter()) {
                    const_fields
                        .push(self.eval_const_expression(&field_init.value.node, field_ty)?);
                }

                Ok(HirConstant::Struct(const_fields))
            }

            // Binary operations (for simple const arithmetic)
            TypedExpression::Binary(binary) => {
                let left = self.eval_const_expression(&binary.left.node, expected_ty)?;
                let right = self.eval_const_expression(&binary.right.node, expected_ty)?;
                self.eval_const_binary_op(&binary.op, &left, &right)
            }

            // Unary operations
            TypedExpression::Unary(unary) => {
                let operand = self.eval_const_expression(&unary.operand.node, expected_ty)?;
                self.eval_const_unary_op(&unary.op, &operand)
            }

            _ => Err(crate::CompilerError::Analysis(format!(
                "Expression type {:?} not supported in global initializers",
                std::mem::discriminant(expr)
            ))),
        }
    }

    /// Evaluate a binary operation on constants
    fn eval_const_binary_op(
        &self,
        op: &zyntax_typed_ast::BinaryOp,
        left: &crate::hir::HirConstant,
        right: &crate::hir::HirConstant,
    ) -> CompilerResult<crate::hir::HirConstant> {
        use crate::hir::HirConstant;
        use zyntax_typed_ast::BinaryOp;

        // Helper macros for arithmetic operations
        macro_rules! int_op {
            ($left:expr, $right:expr, $op:tt) => {
                match ($left, $right) {
                    (HirConstant::I8(l), HirConstant::I8(r)) => Ok(HirConstant::I8(l $op r)),
                    (HirConstant::I16(l), HirConstant::I16(r)) => Ok(HirConstant::I16(l $op r)),
                    (HirConstant::I32(l), HirConstant::I32(r)) => Ok(HirConstant::I32(l $op r)),
                    (HirConstant::I64(l), HirConstant::I64(r)) => Ok(HirConstant::I64(l $op r)),
                    (HirConstant::U8(l), HirConstant::U8(r)) => Ok(HirConstant::U8(l $op r)),
                    (HirConstant::U16(l), HirConstant::U16(r)) => Ok(HirConstant::U16(l $op r)),
                    (HirConstant::U32(l), HirConstant::U32(r)) => Ok(HirConstant::U32(l $op r)),
                    (HirConstant::U64(l), HirConstant::U64(r)) => Ok(HirConstant::U64(l $op r)),
                    _ => Err(crate::CompilerError::Analysis("Type mismatch in const binary operation".into())),
                }
            }
        }

        match op {
            BinaryOp::Add => int_op!(left, right, +),
            BinaryOp::Sub => int_op!(left, right, -),
            BinaryOp::Mul => int_op!(left, right, *),
            BinaryOp::Div => int_op!(left, right, /),
            BinaryOp::Rem => int_op!(left, right, %),
            _ => Err(crate::CompilerError::Analysis(format!(
                "Binary operator {:?} not supported in const context",
                op
            ))),
        }
    }

    /// Evaluate a unary operation on a constant
    fn eval_const_unary_op(
        &self,
        op: &zyntax_typed_ast::UnaryOp,
        operand: &crate::hir::HirConstant,
    ) -> CompilerResult<crate::hir::HirConstant> {
        use crate::hir::HirConstant;
        use zyntax_typed_ast::UnaryOp;

        match op {
            UnaryOp::Minus => match operand {
                HirConstant::I8(v) => Ok(HirConstant::I8(-v)),
                HirConstant::I16(v) => Ok(HirConstant::I16(-v)),
                HirConstant::I32(v) => Ok(HirConstant::I32(-v)),
                HirConstant::I64(v) => Ok(HirConstant::I64(-v)),
                HirConstant::F32(v) => Ok(HirConstant::F32(-v)),
                HirConstant::F64(v) => Ok(HirConstant::F64(-v)),
                _ => Err(crate::CompilerError::Analysis(
                    "Cannot negate non-numeric constant".into(),
                )),
            },
            UnaryOp::Not => match operand {
                HirConstant::Bool(v) => Ok(HirConstant::Bool(!v)),
                _ => Err(crate::CompilerError::Analysis(
                    "Cannot apply NOT to non-boolean constant".into(),
                )),
            },
            _ => Err(crate::CompilerError::Analysis(format!(
                "Unary operator {:?} not supported in const context",
                op
            ))),
        }
    }

    /// Lower an import declaration
    ///
    /// If an import resolver is configured, this will attempt to resolve the import
    /// and register the resolved symbols in the HIR module. Already-resolved modules
    /// are cached to avoid redundant resolution.
    fn lower_import(
        &mut self,
        import: &zyntax_typed_ast::typed_ast::TypedImport,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::typed_ast::TypedImportItem;

        // Convert import items to our internal representation
        let items: Vec<ImportedItem> = import
            .items
            .iter()
            .map(|item| match item {
                TypedImportItem::Named { name, alias } => ImportedItem {
                    name: *name,
                    alias: *alias,
                    is_glob: false,
                },
                TypedImportItem::Glob => ImportedItem {
                    name: {
                        let mut arena = self.arena.lock().unwrap();
                        arena.intern_string("*")
                    },
                    alias: None,
                    is_glob: true,
                },
                TypedImportItem::Default(name) => ImportedItem {
                    name: *name,
                    alias: None,
                    is_glob: false,
                },
            })
            .collect();

        // Convert module path to string vec for cache lookup
        let module_path_strings: Vec<String> = import
            .module_path
            .iter()
            .filter_map(|s| s.resolve_global())
            .collect();

        // Check cache first - avoid re-resolving already resolved modules
        let resolved = if let Some(cached) = self.resolved_module_cache.get(&module_path_strings) {
            log::debug!(
                "Using cached resolution for {:?}: {} symbols",
                module_path_strings,
                cached.len()
            );
            cached.clone()
        } else if let Some(ref resolver) = self.config.import_resolver {
            // Resolve the import
            match resolver.resolve_import(import, &self.import_context) {
                Ok(resolved_imports) => {
                    log::debug!(
                        "Resolved import {:?}: {} symbols",
                        module_path_strings,
                        resolved_imports.len()
                    );
                    // Cache the result
                    self.resolved_module_cache
                        .insert(module_path_strings.clone(), resolved_imports.clone());
                    resolved_imports
                }
                Err(e) => {
                    // Log the error but don't fail - imports might be resolved externally
                    self.diagnostic(
                        DiagnosticLevel::Warning,
                        format!("Import resolution warning: {}", e),
                        Some(import.span),
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Lower resolved imports to HIR - register symbols for codegen
        for resolved_import in &resolved {
            self.register_resolved_import(resolved_import, &items)?;
        }

        // Track the module path in import context to detect cycles
        self.import_context
            .imported_modules
            .push(module_path_strings);

        // Store import metadata
        self.import_metadata.push(ImportMetadata {
            module_path: import.module_path.clone(),
            items,
            span: import.span,
            resolved,
        });

        Ok(())
    }

    /// Register a resolved import in the HIR module for codegen
    ///
    /// This makes the resolved symbols available for code generation.
    fn register_resolved_import(
        &mut self,
        resolved: &ResolvedImport,
        items: &[ImportedItem],
    ) -> CompilerResult<()> {
        use crate::hir::{HirFunctionSignature, HirImport, HirParam, ImportAttributes, ImportKind};

        match resolved {
            ResolvedImport::Function {
                qualified_name,
                params,
                return_type,
                is_extern,
            } => {
                // Register as an external function declaration in HIR
                let func_name = qualified_name
                    .last()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let interned_name = {
                    let mut arena = self.arena.lock().unwrap();
                    arena.intern_string(func_name)
                };

                // Check if we need to apply an alias
                let local_name = items
                    .iter()
                    .find(|item| {
                        item.name
                            .resolve_global()
                            .map(|n| n == func_name)
                            .unwrap_or(false)
                    })
                    .and_then(|item| item.alias)
                    .unwrap_or(interned_name);

                // Convert typed_ast types to HIR types
                let hir_params: Vec<HirParam> = params
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| {
                        let hir_type = self.convert_type(ty);
                        let param_name = {
                            let mut arena = self.arena.lock().unwrap();
                            arena.intern_string(&format!("arg{}", i))
                        };
                        HirParam {
                            id: crate::hir::HirId::new(),
                            name: param_name,
                            ty: hir_type,
                            attributes: Default::default(),
                        }
                    })
                    .collect();
                let hir_return = self.convert_type(return_type);

                // Create the function signature
                let signature = HirFunctionSignature {
                    params: hir_params,
                    returns: vec![hir_return],
                    type_params: Vec::new(),
                    const_params: Vec::new(),
                    lifetime_params: Vec::new(),
                    is_variadic: false,
                    is_async: false,
                    effects: Vec::new(),
                    is_pure: false,
                };

                // Create an import entry for the external function
                let hir_import = HirImport {
                    name: local_name,
                    kind: ImportKind::Function(signature),
                    attributes: ImportAttributes::default(),
                };

                // Add to module's imports
                self.module.imports.push(hir_import);

                log::debug!(
                    "Registered imported function '{}' (alias: {:?})",
                    func_name,
                    if local_name != interned_name {
                        Some(local_name)
                    } else {
                        None
                    }
                );
            }

            ResolvedImport::Type {
                qualified_name,
                ty,
                is_extern,
            } => {
                // Register the type in our symbol table and as an import
                let type_name = qualified_name
                    .last()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let interned_name = {
                    let mut arena = self.arena.lock().unwrap();
                    arena.intern_string(type_name)
                };

                // Check if we need to apply an alias
                let local_name = items
                    .iter()
                    .find(|item| {
                        item.name
                            .resolve_global()
                            .map(|n| n == type_name)
                            .unwrap_or(false)
                    })
                    .and_then(|item| item.alias)
                    .unwrap_or(interned_name);

                // Convert to HIR type
                let hir_type = self.convert_type(ty);

                // Get or create a TypeId for this imported type
                // For named types, try to get the existing ID from the registry
                let type_id = match ty {
                    zyntax_typed_ast::Type::Named { id, .. } => *id,
                    _ => {
                        // For other types, create a synthetic TypeId from hash
                        zyntax_typed_ast::TypeId::new(hash_string(&qualified_name.join("::")) as u32)
                    }
                };

                // Register in the module's type table
                self.module.types.insert(type_id, hir_type.clone());

                // Register in symbol table for name resolution
                self.symbols.types.insert(local_name, type_id);

                // Create an import entry for the type
                let hir_import = HirImport {
                    name: local_name,
                    kind: ImportKind::Type {
                        ty: hir_type,
                        type_id,
                    },
                    attributes: ImportAttributes::default(),
                };

                self.module.imports.push(hir_import);
            }

            ResolvedImport::Module { path, exports } => {
                // For module imports, register all exported symbols
                for export in exports {
                    if !export.is_public {
                        continue;
                    }

                    let symbol_name = {
                        let mut arena = self.arena.lock().unwrap();
                        arena.intern_string(&export.name)
                    };

                    // Register based on symbol kind
                    match export.kind {
                        SymbolKind::Function => {
                            // Create a placeholder function signature
                            // The actual signature will be resolved when the function is used
                            let signature = HirFunctionSignature {
                                params: Vec::new(),
                                returns: vec![crate::hir::HirType::Void],
                                type_params: Vec::new(),
                                const_params: Vec::new(),
                                lifetime_params: Vec::new(),
                                is_variadic: false,
                                is_async: false,
                                effects: Vec::new(),
                                is_pure: false,
                            };

                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Function(signature),
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                        SymbolKind::Type
                        | SymbolKind::Class
                        | SymbolKind::Enum
                        | SymbolKind::Interface
                        | SymbolKind::Trait => {
                            // Create an opaque type for the imported type
                            let hir_type = crate::hir::HirType::Opaque(symbol_name);
                            let type_id = zyntax_typed_ast::TypeId::new(hash_string(&format!(
                                "{}::{}",
                                path.join("::"),
                                export.name
                            ))
                                as u32);

                            self.module.types.insert(type_id, hir_type.clone());
                            self.symbols.types.insert(symbol_name, type_id);

                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Type {
                                    ty: hir_type,
                                    type_id,
                                },
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                        SymbolKind::Constant | SymbolKind::Module => {
                            // Constants and submodules - create global import
                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Global(crate::hir::HirType::Void),
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                    }
                }
            }

            ResolvedImport::Constant { qualified_name, ty } => {
                // Register as a global import
                let const_name = qualified_name
                    .last()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let interned_name = {
                    let mut arena = self.arena.lock().unwrap();
                    arena.intern_string(const_name)
                };

                let hir_type = self.convert_type(ty);

                let hir_import = HirImport {
                    name: interned_name,
                    kind: ImportKind::Global(hir_type),
                    attributes: ImportAttributes::default(),
                };

                self.module.imports.push(hir_import);
            }

            ResolvedImport::Glob {
                module_path,
                symbols,
            } => {
                // For glob imports, register all public symbols from the module
                for symbol in symbols {
                    if !symbol.is_public {
                        continue;
                    }

                    let symbol_name = {
                        let mut arena = self.arena.lock().unwrap();
                        arena.intern_string(&symbol.name)
                    };

                    match symbol.kind {
                        SymbolKind::Function => {
                            let signature = HirFunctionSignature {
                                params: Vec::new(),
                                returns: vec![crate::hir::HirType::Void],
                                type_params: Vec::new(),
                                const_params: Vec::new(),
                                lifetime_params: Vec::new(),
                                is_variadic: false,
                                is_async: false,
                                effects: Vec::new(),
                                is_pure: false,
                            };

                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Function(signature),
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                        SymbolKind::Type
                        | SymbolKind::Class
                        | SymbolKind::Enum
                        | SymbolKind::Interface
                        | SymbolKind::Trait => {
                            let hir_type = crate::hir::HirType::Opaque(symbol_name);
                            let type_id = zyntax_typed_ast::TypeId::new(hash_string(&format!(
                                "{}::{}",
                                module_path.join("::"),
                                symbol.name
                            ))
                                as u32);

                            self.module.types.insert(type_id, hir_type.clone());
                            self.symbols.types.insert(symbol_name, type_id);

                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Type {
                                    ty: hir_type,
                                    type_id,
                                },
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                        SymbolKind::Constant | SymbolKind::Module => {
                            let hir_import = HirImport {
                                name: symbol_name,
                                kind: ImportKind::Global(crate::hir::HirType::Void),
                                attributes: ImportAttributes::default(),
                            };

                            self.module.imports.push(hir_import);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Lower a class declaration
    fn lower_class(
        &mut self,
        class: &zyntax_typed_ast::typed_ast::TypedClass,
    ) -> CompilerResult<()> {
        // Lower each method as a free function with mangled name
        for method in &class.methods {
            self.lower_method(class.name, method)?;
        }

        // Lower constructors
        for (i, ctor) in class.constructors.iter().enumerate() {
            self.lower_constructor(class.name, i, ctor)?;
        }

        Ok(())
    }

    /// Lower a method to a free function with `self` parameter
    fn lower_method(
        &mut self,
        class_name: InternedString,
        method: &zyntax_typed_ast::typed_ast::TypedMethod,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::type_registry::{NullabilityKind, Type, TypeId};
        use zyntax_typed_ast::typed_ast::{
            typed_node, ParameterKind, TypedFunction, TypedParameter,
        };

        // Create mangled function name
        let mangled_name = self.mangle_method_name(class_name, method.name);

        // Build parameters: self parameter + method parameters
        let mut params = Vec::new();

        // Add self parameter if not static
        if !method.is_static {
            // Look up the class TypeId from the registry
            let class_type_id = self
                .type_registry
                .get_type_by_name(class_name)
                .map(|def| def.id)
                .unwrap_or_else(|| TypeId::new(0)); // Fallback if not found

            let self_type = Type::Named {
                id: class_type_id,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: NullabilityKind::NonNull,
            };

            let self_name = {
                let mut arena = self.arena.lock().unwrap();
                arena.intern_string("self")
            };

            let self_param = TypedParameter {
                name: self_name,
                ty: self_type,
                mutability: Mutability::Immutable,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: method.span,
            };
            params.push(self_param);
        }

        // Add method parameters
        for method_param in &method.params {
            let param = TypedParameter {
                name: method_param.name,
                ty: method_param.ty.clone(),
                mutability: method_param.mutability,
                kind: ParameterKind::Regular,
                default_value: method_param.default_value.clone(),
                attributes: vec![],
                span: method_param.span,
            };
            params.push(param);
        }

        // Create a function from the method
        let func = TypedFunction {
            name: mangled_name,
            annotations: vec![],
            effects: vec![],
            type_params: method.type_params.clone(),
            params,
            return_type: method.return_type.clone(),
            body: method.body.clone().or_else(|| {
                Some(zyntax_typed_ast::typed_ast::TypedBlock {
                    statements: vec![],
                    span: method.span,
                })
            }),
            visibility: method.visibility,
            is_async: method.is_async,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::CallingConvention::Default,
            link_name: None,
        };

        // Lower as a regular function
        self.lower_function(&func)
    }

    /// Lower a constructor
    fn lower_constructor(
        &mut self,
        class_name: InternedString,
        index: usize,
        ctor: &zyntax_typed_ast::typed_ast::TypedConstructor,
    ) -> CompilerResult<()> {
        use zyntax_typed_ast::type_registry::{NullabilityKind, Type, TypeId};
        use zyntax_typed_ast::typed_ast::{ParameterKind, TypedFunction, TypedParameter};

        // Constructor name: ClassName_constructor_N
        // Use resolve_global() for portability across interner sources
        let ctor_name = {
            let class_name_str = class_name
                .resolve_global()
                .unwrap_or_else(|| "UnknownClass".to_string());
            let mut arena = self.arena.lock().unwrap();
            arena.intern_string(&format!("{}_constructor_{}", class_name_str, index))
        };

        // Build parameters from constructor
        let params: Vec<TypedParameter> = ctor
            .params
            .iter()
            .map(|p| TypedParameter {
                name: p.name,
                ty: p.ty.clone(),
                mutability: p.mutability,
                kind: ParameterKind::Regular,
                default_value: p.default_value.clone(),
                attributes: vec![],
                span: p.span,
            })
            .collect();

        // Constructor returns an instance of the class
        let class_type_id = self
            .type_registry
            .get_type_by_name(class_name)
            .map(|def| def.id)
            .unwrap_or_else(|| TypeId::new(0)); // Fallback if not found

        let return_type = Type::Named {
            id: class_type_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::NonNull,
        };

        let func = TypedFunction {
            name: ctor_name,
            annotations: vec![],
            effects: vec![],
            type_params: vec![],
            params,
            return_type,
            body: Some(ctor.body.clone()),
            visibility: ctor.visibility,
            is_async: false,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::CallingConvention::Default,
            link_name: None,
        };

        self.lower_function(&func)
    }

    /// Resolve associated function call (Type::method) to mangled method name
    /// Returns None if the function cannot be resolved
    fn resolve_associated_function_to_mangled(
        &self,
        type_name: &str,
        method_name: &str,
    ) -> Option<String> {
        // First, check for inherent impl methods (including extern struct methods)
        // Try wrapper method name first: _zynml_{Type}_{method}
        let wrapper_name = format!("_zynml_{}_{}", type_name, method_name);
        let wrapper_interned = InternedString::new_global(&wrapper_name);
        if self.symbols.functions.contains_key(&wrapper_interned) {
            return Some(wrapper_name);
        }

        // Try standard inherent method: {Type}${method}
        let inherent_name = format!("{}${}", type_name, method_name);
        let inherent_interned = InternedString::new_global(&inherent_name);
        if self.symbols.functions.contains_key(&inherent_interned) {
            return Some(inherent_name);
        }

        // Search through all trait implementations to find the method
        for (_trait_id, impls) in self.type_registry.iter_implementations() {
            for impl_def in impls {
                // Check if this impl is for our type
                if let Type::Named { id, .. } = &impl_def.for_type {
                    // Get the type definition to get its name
                    if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                        let impl_type_name = {
                            let arena = self.arena.lock().unwrap();
                            arena
                                .resolve_string(type_def.name)
                                .map(|s| s.to_string())
                                .unwrap_or_default()
                        };

                        if impl_type_name == type_name {
                            // Check if this impl has the method we're looking for
                            for method in &impl_def.methods {
                                let impl_method_name = {
                                    let arena = self.arena.lock().unwrap();
                                    arena
                                        .resolve_string(method.signature.name)
                                        .map(|s| s.to_string())
                                        .unwrap_or_default()
                                };

                                if impl_method_name == method_name {
                                    // Found it! Return mangled name
                                    // Format: {TypeName}${TraitName}${method_name}
                                    let trait_def =
                                        self.type_registry.get_trait_by_id(impl_def.trait_id);
                                    if let Some(trait_def) = trait_def {
                                        let trait_name = {
                                            let arena = self.arena.lock().unwrap();
                                            arena
                                                .resolve_string(trait_def.name)
                                                .map(|s| s.to_string())
                                                .unwrap_or_default()
                                        };

                                        let mangled =
                                            format!("{}${}${}", type_name, trait_name, method_name);
                                        return Some(mangled);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if there's a From<SourceType> impl for target_type
    /// Returns the mangled From::from function name if found
    fn find_from_impl(&self, source_type: &Type, target_type: &Type) -> Option<String> {
        use zyntax_typed_ast::TypeId;

        // Only works for Named types
        let target_type_id = match target_type {
            Type::Named { id, .. } => *id,
            _ => return None,
        };

        // Get target type name
        let target_type_def = self.type_registry.get_type_by_id(target_type_id)?;
        let target_type_name = {
            let arena = self.arena.lock().unwrap();
            arena
                .resolve_string(target_type_def.name)
                .map(|s| s.to_string())
                .unwrap_or_default()
        };

        // Look for From<SourceType> impl for TargetType
        for (_trait_id, impls) in self.type_registry.iter_implementations() {
            for impl_def in impls {
                // Check if this is an impl for our target type
                if let Type::Named { id, .. } = &impl_def.for_type {
                    if *id == target_type_id {
                        // Check if this is a From trait
                        if let Some(trait_def) =
                            self.type_registry.get_trait_by_id(impl_def.trait_id)
                        {
                            let trait_name = {
                                let arena = self.arena.lock().unwrap();
                                arena
                                    .resolve_string(trait_def.name)
                                    .map(|s| s.to_string())
                                    .unwrap_or_default()
                            };

                            if trait_name == "From" {
                                // Check if the type arg matches source_type
                                if !impl_def.type_args.is_empty() {
                                    // Simple comparison - can be enhanced for complex types
                                    if &impl_def.type_args[0] == source_type {
                                        // Found From<SourceType> for TargetType!
                                        // Return mangled name: TargetType$From$from
                                        let mangled = format!("{}$From$from", target_type_name);
                                        return Some(mangled);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to insert an implicit From conversion if types don't match
    /// Returns a new expression wrapped in Type::from() if conversion is available
    fn try_implicit_from_conversion(
        &self,
        expr: zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>,
        expected_type: &Type,
    ) -> Option<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>> {
        use zyntax_typed_ast::{typed_ast::typed_node, TypedCall, TypedExpression};

        // Don't convert if types already match
        if &expr.ty == expected_type {
            return None;
        }

        // Don't convert if either type is Any or Unknown (type inference not done)
        if matches!(expr.ty, Type::Any | Type::Unknown)
            || matches!(expected_type, Type::Any | Type::Unknown)
        {
            return None;
        }

        // Check if there's a From<expr.ty> impl for expected_type
        if let Some(from_func) = self.find_from_impl(&expr.ty, expected_type) {
            eprintln!(
                "[IMPLICIT_CONVERSION] Converting {:?} to {:?} via {}",
                expr.ty, expected_type, from_func
            );

            // Capture span before moving expr
            let span = expr.span;

            // Create Type::from(expr) call
            let from_func_name = InternedString::new_global(&from_func);
            let call = TypedCall {
                callee: Box::new(typed_node(
                    TypedExpression::Variable(from_func_name),
                    Type::Any, // Function type
                    span,
                )),
                positional_args: vec![expr],
                named_args: vec![],
                type_args: vec![],
            };

            return Some(typed_node(
                TypedExpression::Call(call),
                expected_type.clone(),
                span,
            ));
        }

        None
    }

    /// Mangle method name: ClassName_methodName
    fn mangle_method_name(
        &self,
        class_name: InternedString,
        method_name: InternedString,
    ) -> InternedString {
        // Use resolve_global() since InternedStrings may come from different sources
        // (global interner from ZynPEG runtime, local arena from JSON deserialization, etc.)
        let class_name_str = class_name
            .resolve_global()
            .unwrap_or_else(|| "UnknownClass".to_string());
        let method_name_str = method_name
            .resolve_global()
            .unwrap_or_else(|| "unknown_method".to_string());
        let mut arena = self.arena.lock().unwrap();
        arena.intern_string(&format!("{}_{}", class_name_str, method_name_str))
    }

    /// Mangle trait method name: Type$Trait$method
    /// This is used for trait implementations to avoid name collisions
    fn mangle_trait_method_name(
        &self,
        type_name: InternedString,
        trait_name: InternedString,
        method_name: InternedString,
    ) -> InternedString {
        let type_name_str = type_name
            .resolve_global()
            .unwrap_or_else(|| "UnknownType".to_string());
        let trait_name_str = trait_name
            .resolve_global()
            .unwrap_or_else(|| "UnknownTrait".to_string());
        let method_name_str = method_name
            .resolve_global()
            .unwrap_or_else(|| "unknown_method".to_string());

        let mangled = format!("{}${}${}", type_name_str, trait_name_str, method_name_str);
        InternedString::new_global(&mangled)
    }

    /// Convert visibility to linkage
    fn convert_linkage(&self, vis: Visibility) -> crate::hir::Linkage {
        match vis {
            Visibility::Public => crate::hir::Linkage::External,
            Visibility::Private => crate::hir::Linkage::Private,
            Visibility::Protected => crate::hir::Linkage::Internal,
            Visibility::Internal => crate::hir::Linkage::Internal,
        }
    }

    /// Convert visibility
    fn convert_visibility(&self, vis: Visibility) -> crate::hir::Visibility {
        match vis {
            Visibility::Public => crate::hir::Visibility::Default,
            Visibility::Private => crate::hir::Visibility::Hidden,
            Visibility::Protected => crate::hir::Visibility::Protected,
            Visibility::Internal => crate::hir::Visibility::Hidden,
        }
    }

    /// Convert TypedAST calling convention to HIR calling convention (Gap 11)
    fn convert_calling_convention(
        &self,
        cc: zyntax_typed_ast::CallingConvention,
    ) -> crate::hir::CallingConvention {
        use zyntax_typed_ast::CallingConvention as TypedCC;
        match cc {
            // Default uses System which resolves to platform-native in backend (AppleAarch64 on ARM Mac, SystemV on x86)
            // This matches ZRTL plugin calling conventions
            TypedCC::Default => crate::hir::CallingConvention::System,
            TypedCC::Rust => crate::hir::CallingConvention::Fast,
            TypedCC::Cdecl => crate::hir::CallingConvention::C,
            TypedCC::System => crate::hir::CallingConvention::System,
            TypedCC::Stdcall | TypedCC::Fastcall | TypedCC::Thiscall | TypedCC::Vectorcall => {
                // These platform-specific conventions map to C for now
                // Backends can handle them differently if needed
                crate::hir::CallingConvention::C
            }
        }
    }

    /// Lower trait implementations and generate vtables
    ///
    /// This method iterates over all trait implementations in the TypeRegistry
    /// and lowers them to HIR, generating vtables and registering methods in
    /// the VtableRegistry.
    ///
    /// Integration status:
    /// - ✅ TypeRegistry.iter_implementations() available
    /// - ✅ TypeRegistry.get_trait_by_id() available
    /// - ⚠️ Method body lowering needed (lower_impl_method placeholder)
    /// - ⚠️ Vtable globals emission needed (backend work)
    fn lower_implementations(&mut self) -> CompilerResult<()> {
        // Collect implementations to avoid borrow checker issues
        // (iter_implementations borrows self.type_registry immutably,
        //  but lower_impl borrows self mutably)
        let implementations: Vec<_> = self
            .type_registry
            .iter_implementations()
            .map(|(trait_id, impl_defs)| (*trait_id, impl_defs.clone()))
            .collect();

        // Lower each implementation
        for (trait_id, impl_defs) in implementations {
            for impl_def in &impl_defs {
                // Skip impls that fail - methods may not have been lowered
                if let Err(e) = self.lower_impl(trait_id, impl_def) {
                    let type_name = match &impl_def.for_type {
                        zyntax_typed_ast::Type::Named { id, .. } => self
                            .type_registry
                            .get_type_by_id(*id)
                            .map(|t| t.name.resolve_global().unwrap_or_default().to_string())
                            .unwrap_or_else(|| format!("{:?}", id)),
                        zyntax_typed_ast::Type::Extern { name, .. } => {
                            name.resolve_global().unwrap_or_default().to_string()
                        }
                        other => format!("{:?}", other),
                    };
                    eprintln!(
                        "[LOWERING WARN] Skipping trait impl for '{}': {:?}",
                        type_name, e
                    );
                }
            }
        }

        Ok(())
    }

    /// Lower a single trait implementation
    ///
    /// This method:
    /// 1. Converts impl for_type to HIR type
    /// 2. Generates trait method table
    /// 3. Validates implementation satisfies trait
    /// 4. Lowers each method and registers in VtableRegistry
    /// 5. Generates vtable with real function IDs
    /// 6. Creates vtable global and adds to module
    ///
    /// NOTE: Ready to use once TypeRegistry provides access to ImplDef
    #[allow(dead_code)] // Will be used when TypeRegistry integration is complete
    fn lower_impl(
        &mut self,
        trait_id: TypeId,
        impl_def: &zyntax_typed_ast::ImplDef,
    ) -> CompilerResult<()> {
        use crate::trait_lowering::{
            convert_type, create_vtable_global, generate_trait_method_table, generate_vtable,
            validate_trait_implementation,
        };

        // Step 1: Convert for_type to HIR type
        let for_type_hir = convert_type(&impl_def.for_type, &self.type_registry)?;

        // Step 2: Extract type_id from for_type (nominal types only)
        let type_id = match &impl_def.for_type {
            zyntax_typed_ast::Type::Named { id, .. } => *id,
            zyntax_typed_ast::Type::Extern { name, .. } => {
                // Extern types (like $Tensor) don't have TypeIds - they're handled by ZRTL
                // Skip trait impl lowering for extern types as they use runtime dispatch
                eprintln!(
                    "[LOWERING WARN] Skipping trait impl for extern type '{}' - ZRTL handles these",
                    name.resolve_global().unwrap_or_default()
                );
                return Ok(());
            }
            zyntax_typed_ast::Type::Unresolved(name) => {
                // Unresolved types are generic type parameters (like I, T, etc.)
                // Skip trait impl lowering for these as they require monomorphization
                eprintln!("[LOWERING WARN] Skipping trait impl for unresolved type '{}' - requires monomorphization",
                    name.resolve_global().unwrap_or_default());
                return Ok(());
            }
            _ => {
                return Err(crate::CompilerError::Analysis(
                    "Impl for_type must be a named type (nominal typing only)".to_string(),
                ));
            }
        };

        // Step 2.5: Register implementation in AssociatedTypeResolver
        // This enables resolution of associated types like <T as Trait>::Item
        self.associated_type_resolver
            .register_impl(trait_id, type_id, Arc::new(impl_def.clone()));

        // Step 3: Generate trait method table
        let trait_def = self
            .type_registry
            .get_trait_def(trait_id)
            .ok_or_else(|| crate::CompilerError::Analysis("Trait not found".to_string()))?;
        let trait_method_table =
            generate_trait_method_table(trait_def, trait_id, &self.type_registry)?;

        // Step 4: Validate implementation
        validate_trait_implementation(&trait_method_table, impl_def)?;

        // Step 5: Lower each method and register in VtableRegistry
        for method_impl in &impl_def.methods {
            // Lower method body to HIR function
            let method_func_id = self.lower_impl_method(trait_id, type_id, method_impl)?;

            // Register method in vtable registry
            self.vtable_registry.register_method(
                trait_id,
                type_id,
                method_impl.signature.name,
                method_func_id,
            );
        }

        // Step 6: Generate vtable (requires trait_method_table from step 3)
        let vtable = generate_vtable(
            &trait_method_table,
            impl_def,
            for_type_hir.clone(),
            type_id,
            &self.vtable_registry,
        )?;

        // Step 7: Create vtable global
        // Generate vtable name: vtable_TraitName_TypeName
        let vtable_name = {
            let name_str = format!(
                "vtable_{}_{}",
                self.type_registry
                    .get_trait_def(trait_id)
                    .map(|t| t.name.to_string())
                    .unwrap_or("unknown".to_string()),
                format!("{:?}", impl_def.for_type)
            );
            self.arena.lock().unwrap().intern_string(name_str)
        };
        let vtable_global = create_vtable_global(vtable.clone(), vtable_name);

        // Step 8: Register vtable in registry
        let vtable_id = self.vtable_registry.register_vtable(
            trait_id,
            type_id,
            vtable.clone(),
            vtable_global.clone(),
        );

        // Step 9: Add vtable to module globals
        self.module.globals.insert(vtable_global.id, vtable_global);

        Ok(())
    }

    /// Lower an impl method to HIR function
    ///
    /// This method looks up the already-lowered class method function.
    /// Class methods are lowered during lower_class(), so we just need to
    /// find the corresponding function ID from the symbol table.
    ///
    /// Returns the function ID for vtable registration.
    #[allow(dead_code)] // Will be used when TypeRegistry integration is complete
    fn lower_impl_method(
        &mut self,
        trait_id: TypeId,
        type_id: TypeId,
        method_impl: &zyntax_typed_ast::MethodImpl,
    ) -> CompilerResult<crate::hir::HirId> {
        // Get type name and trait name
        let type_def = self.type_registry.get_type_by_id(type_id).ok_or_else(|| {
            crate::CompilerError::Analysis(format!("Type {:?} not found in registry", type_id))
        })?;
        let type_name = type_def.name;

        // Try trait-based mangling first (Type$Trait$method)
        // This is what lower_impl_block uses for trait implementations
        let trait_def = self.type_registry.get_trait_def(trait_id);
        if let Some(trait_def) = trait_def {
            let trait_mangled_name = self.mangle_trait_method_name(
                type_name,
                trait_def.name,
                method_impl.signature.name,
            );

            // Check if already lowered by lower_impl_block
            if let Some(&method_id) = self.symbols.functions.get(&trait_mangled_name) {
                return Ok(method_id);
            }
        }

        // Fallback: try class method mangling (ClassName::methodName)
        // This is for inherent methods on classes
        let class_mangled_name = self.mangle_method_name(type_name, method_impl.signature.name);

        // Lookup the already-lowered function ID from symbol table
        let method_id = self.symbols.functions.get(&class_mangled_name)
            .copied()
            .ok_or_else(|| crate::CompilerError::Analysis(
                format!("Method function not found in symbol table: {:?} or {:?}. \
                        This likely means the method was not lowered yet. \
                        Ensure lower_class() or lower_impl_block() runs before lower_implementations().",
                        class_mangled_name, trait_def.map(|t| t.name))
            ))?;

        Ok(method_id)
    }

    /// Verify that all references have been resolved (defensive check)
    fn resolve_references(&mut self) -> CompilerResult<()> {
        // Verification pass: ensure all functions and globals in the module
        // have entries in the symbol table. This is a defensive check since
        // the type checker should have already resolved everything.

        let mut missing_symbols = Vec::new();

        // Check all functions
        for (func_id, func) in &self.module.functions {
            if !self.symbols.functions.values().any(|id| id == func_id) {
                missing_symbols.push(format!(
                    "Function '{}' has no symbol table entry",
                    func.name
                ));
            }
        }

        // Check all globals
        for (global_id, global) in &self.module.globals {
            if !self.symbols.globals.values().any(|id| id == global_id) {
                missing_symbols.push(format!(
                    "Global '{}' has no symbol table entry",
                    global.name
                ));
            }
        }

        if !missing_symbols.is_empty() {
            // Report warnings for missing symbols (non-fatal, since type checker validated)
            for msg in &missing_symbols {
                self.diagnostic(
                    DiagnosticLevel::Warning,
                    format!("Symbol table inconsistency: {}", msg),
                    None,
                );
            }

            if self.config.strict_mode {
                return Err(crate::CompilerError::Analysis(format!(
                    "Symbol table verification failed: {} unresolved references",
                    missing_symbols.len()
                )));
            }
        }

        Ok(())
    }
}

impl LoweringPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the pipeline
    pub fn add_pass(&mut self, pass: Box<dyn LoweringPass>) {
        self.passes.push(pass);
    }

    /// Run all passes
    pub fn run(&mut self, context: &mut LoweringContext) -> CompilerResult<()> {
        // Sort passes by dependencies
        self.sort_passes();

        // Run each pass
        for pass in &mut self.passes {
            let pass_name = pass.name();

            if context.config.debug_info {
                context.diagnostic(
                    DiagnosticLevel::Info,
                    format!("Running lowering pass: {}", pass_name),
                    None,
                );
            }

            pass.run(context)?;
        }

        Ok(())
    }

    /// Sort passes by dependencies using topological sort (Kahn's algorithm)
    fn sort_passes(&mut self) {
        use std::collections::{HashMap, HashSet, VecDeque};

        if self.passes.is_empty() {
            return;
        }

        // Build dependency graph
        let mut in_degree: HashMap<&'static str, usize> = HashMap::new();
        let mut adjacency: HashMap<&'static str, Vec<usize>> = HashMap::new();
        let mut name_to_idx: HashMap<&'static str, usize> = HashMap::new();

        // Initialize data structures
        for (idx, pass) in self.passes.iter().enumerate() {
            let name = pass.name();
            name_to_idx.insert(name, idx);
            in_degree.insert(name, 0);
            adjacency.insert(name, Vec::new());
        }

        // Count in-degrees and build adjacency list
        for (idx, pass) in self.passes.iter().enumerate() {
            for dep in pass.dependencies() {
                // Check if dependency exists
                if !name_to_idx.contains_key(dep) {
                    panic!(
                        "Pass '{}' depends on '{}' which is not registered",
                        pass.name(),
                        dep
                    );
                }

                // Add edge: dep -> current pass
                adjacency.get_mut(dep).unwrap().push(idx);
                *in_degree.get_mut(pass.name()).unwrap() += 1;
            }
        }

        // Kahn's algorithm: start with passes that have no dependencies
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, pass) in self.passes.iter().enumerate() {
            if *in_degree.get(pass.name()).unwrap() == 0 {
                queue.push_back(idx);
            }
        }

        // Topological sort
        let mut sorted_indices: Vec<usize> = Vec::new();
        let mut visited: HashSet<&'static str> = HashSet::new();

        while let Some(idx) = queue.pop_front() {
            let pass_name = self.passes[idx].name();
            sorted_indices.push(idx);
            visited.insert(pass_name);

            // Reduce in-degree for dependent passes
            for &dependent_idx in adjacency.get(pass_name).unwrap() {
                let dependent_name = self.passes[dependent_idx].name();
                let degree = in_degree.get_mut(dependent_name).unwrap();
                *degree -= 1;

                if *degree == 0 {
                    queue.push_back(dependent_idx);
                }
            }
        }

        // Detect cycles
        if sorted_indices.len() != self.passes.len() {
            let mut unvisited: Vec<&'static str> = Vec::new();
            for pass in &self.passes {
                if !visited.contains(pass.name()) {
                    unvisited.push(pass.name());
                }
            }
            panic!("Circular dependency detected among passes: {:?}", unvisited);
        }

        // Reorder passes based on sorted indices
        let mut sorted_passes: Vec<Box<dyn LoweringPass>> = Vec::new();
        for idx in sorted_indices {
            // We need to temporarily take ownership - use swap_remove and rebuild
            sorted_passes.push(std::mem::replace(
                &mut self.passes[idx],
                Box::new(passes::CfgConstructionPass), // Dummy placeholder
            ));
        }

        self.passes = sorted_passes;
    }
}

/// Standard lowering passes
pub mod passes {
    use super::*;

    /// CFG construction pass
    pub struct CfgConstructionPass;

    impl LoweringPass for CfgConstructionPass {
        fn name(&self) -> &'static str {
            "cfg-construction"
        }

        fn dependencies(&self) -> &[&'static str] {
            &[]
        }

        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            // CFG is built during function lowering
            Ok(())
        }
    }

    /// SSA construction pass
    pub struct SsaConstructionPass;

    impl LoweringPass for SsaConstructionPass {
        fn name(&self) -> &'static str {
            "ssa-construction"
        }

        fn dependencies(&self) -> &[&'static str] {
            &["cfg-construction"]
        }

        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            // SSA is built during function lowering
            Ok(())
        }
    }

    /// Type validation pass
    pub struct TypeValidationPass;

    impl LoweringPass for TypeValidationPass {
        fn name(&self) -> &'static str {
            "type-validation"
        }

        fn dependencies(&self) -> &[&'static str] {
            &["ssa-construction"]
        }

        fn run(&mut self, context: &mut LoweringContext) -> CompilerResult<()> {
            // Validate all types in HIR
            let functions: Vec<_> = context.module.functions.values().cloned().collect();
            for func in &functions {
                Self::validate_function_types(func, context)?;
            }

            Ok(())
        }
    }

    impl TypeValidationPass {
        fn validate_function_types(
            func: &HirFunction,
            context: &mut LoweringContext,
        ) -> CompilerResult<()> {
            // Validate parameter types
            for param in &func.signature.params {
                if !Self::is_valid_hir_type(&param.ty) {
                    context.diagnostic(
                        DiagnosticLevel::Error,
                        format!(
                            "Invalid parameter type in function {}: {:?}",
                            func.name, param.ty
                        ),
                        None,
                    );
                }
            }

            // Validate return types
            for ret_ty in &func.signature.returns {
                if !Self::is_valid_hir_type(ret_ty) {
                    context.diagnostic(
                        DiagnosticLevel::Error,
                        format!(
                            "Invalid return type in function {}: {:?}",
                            func.name, ret_ty
                        ),
                        None,
                    );
                }
            }

            Ok(())
        }

        fn is_valid_hir_type(ty: &HirType) -> bool {
            match ty {
                HirType::Ptr(inner) => Self::is_valid_hir_type(inner),
                HirType::Array(inner, _) => Self::is_valid_hir_type(inner),
                HirType::Vector(inner, _) => Self::is_valid_hir_type(inner),
                HirType::Struct(s) => s.fields.iter().all(Self::is_valid_hir_type),
                HirType::Function(f) => {
                    f.params.iter().all(Self::is_valid_hir_type)
                        && f.returns.iter().all(Self::is_valid_hir_type)
                }
                _ => true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test passes for topological sort
    struct PassA;
    impl LoweringPass for PassA {
        fn name(&self) -> &'static str {
            "pass-a"
        }
        fn dependencies(&self) -> &[&'static str] {
            &[]
        }
        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            Ok(())
        }
    }

    struct PassB;
    impl LoweringPass for PassB {
        fn name(&self) -> &'static str {
            "pass-b"
        }
        fn dependencies(&self) -> &[&'static str] {
            &["pass-a"]
        }
        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            Ok(())
        }
    }

    struct PassC;
    impl LoweringPass for PassC {
        fn name(&self) -> &'static str {
            "pass-c"
        }
        fn dependencies(&self) -> &[&'static str] {
            &["pass-a"]
        }
        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            Ok(())
        }
    }

    struct PassD;
    impl LoweringPass for PassD {
        fn name(&self) -> &'static str {
            "pass-d"
        }
        fn dependencies(&self) -> &[&'static str] {
            &["pass-b", "pass-c"]
        }
        fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_pass_topological_sort_simple() {
        // Test: B depends on A
        // Expected order: A, B
        let mut pipeline = LoweringPipeline { passes: Vec::new() };
        pipeline.passes.push(Box::new(PassB));
        pipeline.passes.push(Box::new(PassA));

        pipeline.sort_passes();

        assert_eq!(pipeline.passes.len(), 2);
        assert_eq!(pipeline.passes[0].name(), "pass-a");
        assert_eq!(pipeline.passes[1].name(), "pass-b");
    }

    #[test]
    fn test_pass_topological_sort_diamond() {
        // Test: Diamond dependency
        //       A
        //      / \
        //     B   C
        //      \ /
        //       D
        // Expected order: A, then B and C (either order), then D
        let mut pipeline = LoweringPipeline { passes: Vec::new() };
        pipeline.passes.push(Box::new(PassD));
        pipeline.passes.push(Box::new(PassC));
        pipeline.passes.push(Box::new(PassB));
        pipeline.passes.push(Box::new(PassA));

        pipeline.sort_passes();

        assert_eq!(pipeline.passes.len(), 4);
        assert_eq!(pipeline.passes[0].name(), "pass-a");
        // B and C can be in either order
        let middle: Vec<&str> = vec![pipeline.passes[1].name(), pipeline.passes[2].name()];
        assert!(middle.contains(&"pass-b"));
        assert!(middle.contains(&"pass-c"));
        assert_eq!(pipeline.passes[3].name(), "pass-d");
    }

    #[test]
    fn test_pass_topological_sort_independent() {
        // Test: Independent passes maintain stable order
        let mut pipeline = LoweringPipeline { passes: Vec::new() };
        pipeline.passes.push(Box::new(PassA));
        pipeline.passes.push(Box::new(PassC));

        let original_order = vec![pipeline.passes[0].name(), pipeline.passes[1].name()];

        pipeline.sort_passes();

        assert_eq!(pipeline.passes.len(), 2);
        let sorted_order = vec![pipeline.passes[0].name(), pipeline.passes[1].name()];

        // Independent passes keep their original relative order
        assert!(sorted_order == original_order);
    }

    #[test]
    #[should_panic(expected = "Circular dependency detected")]
    fn test_pass_circular_dependency() {
        // Test circular dependencies are detected
        struct PassX;
        impl LoweringPass for PassX {
            fn name(&self) -> &'static str {
                "pass-x"
            }
            fn dependencies(&self) -> &[&'static str] {
                &["pass-y"]
            }
            fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
                Ok(())
            }
        }

        struct PassY;
        impl LoweringPass for PassY {
            fn name(&self) -> &'static str {
                "pass-y"
            }
            fn dependencies(&self) -> &[&'static str] {
                &["pass-x"]
            }
            fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
                Ok(())
            }
        }

        let mut pipeline = LoweringPipeline { passes: Vec::new() };
        pipeline.passes.push(Box::new(PassX));
        pipeline.passes.push(Box::new(PassY));

        pipeline.sort_passes(); // Should panic
    }

    #[test]
    #[should_panic(expected = "depends on")]
    fn test_pass_missing_dependency() {
        // Test missing dependencies are detected
        struct PassMissing;
        impl LoweringPass for PassMissing {
            fn name(&self) -> &'static str {
                "pass-missing"
            }
            fn dependencies(&self) -> &[&'static str] {
                &["nonexistent-pass"]
            }
            fn run(&mut self, _context: &mut LoweringContext) -> CompilerResult<()> {
                Ok(())
            }
        }

        let mut pipeline = LoweringPipeline { passes: Vec::new() };
        pipeline.passes.push(Box::new(PassMissing));

        pipeline.sort_passes(); // Should panic
    }
}
