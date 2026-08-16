//! The classic runtime: compile a program, then call into it.
//!
//! One Cranelift backend, one symbol table, and the external functions
//! a host registers for the compiled code to call back out through.

use super::native_call::{
    call_dynamic_function, call_native_with_signature, call_with_signature, dynamic_to_i64,
};
use super::promise::ZyntaxPromise;
use super::types::{
    CompiledImportResolverCallback, ImportResolverCallback, NativeSignature, NativeType,
    RuntimeError, RuntimeEvent, RuntimeResult,
};
use super::{
    apply_krio_async_lowering, apply_krio_effect_lowering, apply_krio_fiber_lowering,
    capture_runtime_events_from_program, export_conflicts, synthesize_handler_state,
};
use crate::convert::FromZyntax;
use crate::error::ZyntaxError;
use crate::grammar::{GrammarError, LanguageGrammar};
use crate::value::ZyntaxValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    cranelift_backend::CraneliftBackend,
    hir::{HirId, HirModule},
    lowering::AstLowering,
    runtime::{Executor, Waker as RuntimeWaker},
    zrtl::DynamicValue,
    CompilationConfig, CompilerError,
};

pub struct ZyntaxRuntime {
    /// The Cranelift JIT backend
    backend: CraneliftBackend,
    /// Mapping from function names to HIR IDs
    function_ids: HashMap<String, HirId>,
    /// Mapping from function names to their native signatures
    function_signatures: HashMap<String, NativeSignature>,
    /// Compilation configuration
    config: CompilationConfig,
    /// Registered external functions
    external_functions: HashMap<String, ExternalFunction>,
    /// What a grammar's builtin names stand for, as names rather than
    /// as the symbols they resolve to. Kept unresolved so a plugin can
    /// load before or after the grammar that names it.
    builtin_aliases: HashMap<String, String>,
    /// Import resolver callbacks (tried in order)
    import_resolvers: Vec<ImportResolverCallback>,
    /// Build-time parsed imports, consulted before source resolvers.
    compiled_import_resolvers: Vec<CompiledImportResolverCallback>,
    /// Modules a snapshot installed, keyed by the language that
    /// brought them, so a name means what it means inside the language
    /// asking rather than whichever language registered first.
    snapshot_modules: crate::import_chain::SnapshotModules,
    /// Which language exported each symbol, for the ones loaded as a
    /// module of a named language. A symbol name is shared across
    /// every language in a runtime, so this is what makes a second
    /// language taking one an error rather than an overwrite.
    exported_by: HashMap<String, String>,
    /// Registered language grammars (language name -> grammar)
    grammars: HashMap<String, Arc<LanguageGrammar>>,
    /// File extension to language mapping (e.g., ".zig" -> "zig")
    extension_map: HashMap<String, String>,
    /// Names of async functions (original name, not _new suffix)
    async_functions: std::collections::HashSet<String>,
    /// Plugin signatures (symbol name -> ZRTL signature)
    /// Collected from loaded plugins for proper extern function type checking
    plugin_signatures: HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    /// Plugins whose symbols have been registered.
    ///
    /// Registration hands out raw pointers into the plugin's library, so the
    /// library has to outlive them. Dropping it closes the handle, and on a
    /// platform where that actually unmaps the image every registered
    /// pointer is left dangling.
    loaded_plugins: Vec<zyntax_compiler::zrtl::ZrtlPlugin>,
    /// Captured runtime semantic events (render/stream).
    runtime_events: Vec<RuntimeEvent>,
    /// Optional callback invoked whenever a runtime event is captured.
    event_sink: Option<Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
    /// The single BC interpreter that owns execution dispatch. Tier-up
    /// to Cranelift / LLVM happens inside it via beadie's
    /// `TieredAdapter` — there is no parallel beadie loop on the
    /// `ZyntaxRuntime` side. `compile_module` installs the same HIR
    /// here that the legacy JIT receives; `call_function` delegates
    /// to it. Wrapped in `Mutex` so `&self` methods can mutate the
    /// interp's per-call state.
    interp: std::sync::Mutex<crate::interp_runtime::InterpRuntime>,
    /// Wrapper-class registry for compiler-known built-in types
    /// (`Fiber<T>`, future `SimdVector<T, N>`, ...). Seeded with the
    /// compiler's defaults at construction; embedders register
    /// additional classes via `register_builtin_class` BEFORE any
    /// compilation runs. Wrapped in `Arc<Mutex<_>>` so the
    /// registration API can mutate while compilation reads a clone
    /// of the `Arc<BuiltinRegistry>` snapshot.
    builtin_registry: Arc<std::sync::Mutex<zyntax_compiler::builtin_class::BuiltinRegistry>>,
}

/// An external function that can be called from Zyntax code
#[derive(Clone)]
pub struct ExternalFunction {
    /// Function name
    pub name: String,
    /// Function pointer
    pub ptr: *const u8,
    /// Expected argument count
    pub arg_count: usize,
}

// SAFETY: Function pointers are inherently thread-unsafe, but we manage
// access through the runtime's mutex-protected state
unsafe impl Send for ExternalFunction {}
unsafe impl Sync for ExternalFunction {}

impl ZyntaxRuntime {
    /// Create a new runtime with default configuration
    pub fn new() -> RuntimeResult<Self> {
        Self::with_config(CompilationConfig::default())
    }

    /// Create a new runtime with custom configuration
    pub fn with_config(config: CompilationConfig) -> RuntimeResult<Self> {
        let mut backend = CraneliftBackend::new()?;

        // OSR back-edge probes fire at every loop header but the tier-up
        // consumer side isn't wired on this path. Until it is, they are
        // pure overhead. The TieredRuntime::new path (line ~2933) and
        // install_interp_jit_with (interp_runtime.rs:708) already gate
        // probes off the same way; this site was missed when commits
        // 5354662 / d81a7d6 landed, leaving ZyntaxRuntime consumers
        // (bench harness, zynml CLI) paying ~22% CPU on probe tick + sample
        // even though the consumer never reads them.
        backend.set_emit_osr_probes(false);

        let mut runtime = Self {
            backend,
            function_ids: HashMap::new(),
            function_signatures: HashMap::new(),
            config,
            external_functions: HashMap::new(),
            builtin_aliases: HashMap::new(),
            import_resolvers: Vec::new(),
            compiled_import_resolvers: Vec::new(),
            snapshot_modules: Default::default(),
            exported_by: HashMap::new(),
            grammars: HashMap::new(),
            extension_map: HashMap::new(),
            async_functions: std::collections::HashSet::new(),
            plugin_signatures: HashMap::new(),
            loaded_plugins: Vec::new(),
            runtime_events: Vec::new(),
            event_sink: None,
            interp: std::sync::Mutex::new(crate::interp_runtime::InterpRuntime::new()),
            builtin_registry: Arc::new(std::sync::Mutex::new(
                zyntax_compiler::builtin_class::BuiltinRegistry::with_defaults(),
            )),
        };
        // Register the algebraic-effects runtime symbols
        // (`__zyntax_effect_*`) up front so any module compiled later
        // that references them links cleanly. No-op for modules that
        // don't (the JIT only resolves symbols the IR actually calls).
        crate::effect_runtime::register_effect_runtime_symbols(&mut runtime);
        // Same idea for the `Type::Any` autobox / autounbox helpers.
        // SSA lowering emits `Call::Symbol("zyntax_box_X")` /
        // `..._get_X` for any module that stores into or reads from
        // a `Type::Any` field — register them up front with typed
        // signatures so the JIT picks the right param/return shape.
        crate::effect_runtime::register_box_runtime_symbols(&mut runtime);
        // First-class fiber HIR ops lower to `Call::Symbol("krio_fiber_*")`
        // before the backend sees them. Register the typed signatures up
        // front so link resolution is clean even when no fiber op fires,
        // and install the default krio-fiber backed `FiberCfg` so that
        // any op that does fire reaches a real implementation rather
        // than the panic stub. `install` is set-once at the process
        // level; subsequent runtimes share the same backend.
        crate::effect_runtime::register_fiber_runtime_symbols(&mut runtime);
        let _ = krio_adapter::fiber::install();
        runtime.finalize_runtime_symbols()?;
        Ok(runtime)
    }

    /// Create a new runtime with additional runtime symbols for FFI
    ///
    /// This allows linking external C functions or Rust functions into the JIT.
    pub fn with_symbols(symbols: &[(&str, *const u8)]) -> RuntimeResult<Self> {
        let backend = CraneliftBackend::with_runtime_symbols(symbols)?;

        let mut runtime = Self {
            backend,
            function_ids: HashMap::new(),
            function_signatures: HashMap::new(),
            config: CompilationConfig::default(),
            external_functions: HashMap::new(),
            builtin_aliases: HashMap::new(),
            import_resolvers: Vec::new(),
            compiled_import_resolvers: Vec::new(),
            snapshot_modules: Default::default(),
            exported_by: HashMap::new(),
            grammars: HashMap::new(),
            extension_map: HashMap::new(),
            async_functions: std::collections::HashSet::new(),
            plugin_signatures: HashMap::new(),
            loaded_plugins: Vec::new(),
            runtime_events: Vec::new(),
            event_sink: None,
            interp: std::sync::Mutex::new(crate::interp_runtime::InterpRuntime::new()),
            builtin_registry: Arc::new(std::sync::Mutex::new(
                zyntax_compiler::builtin_class::BuiltinRegistry::with_defaults(),
            )),
        };
        crate::effect_runtime::register_effect_runtime_symbols(&mut runtime);
        for (name, ptr, arity) in zyntax_compiler::zrtl::box_runtime_symbols() {
            runtime.backend.register_runtime_symbol(name, ptr);
            if let Ok(mut interp) = runtime.interp.lock() {
                interp.register_symbol(name.to_string(), ptr, arity);
            }
        }
        runtime.finalize_runtime_symbols()?;
        Ok(runtime)
    }

    /// Compile a HIR module into the runtime
    ///
    /// After compilation, functions can be called via `call()` or `call_async()`.
    ///
    /// If the module has extern declarations that match previously compiled functions,
    /// the backend will be rebuilt to include those symbols before compilation.
    pub fn compile_module(&mut self, module: &zyntax_compiler::HirModule) -> RuntimeResult<()> {
        // Run interp-safe HIR opts before backend installation. Without this,
        // user programs run through `compile_module` never get CSE / LICM /
        // inline / const_fold / aggregate_split — the bench-only
        // `run_interp_safe_opts` entry was the only place these fired,
        // leaving production code unoptimised. (Skippable via
        // `ZYNTAX_DISABLE_INTERP_OPTS=1` for the rare case where we want
        // to bisect against the raw lowered HIR.)
        let mut owned = module.clone();
        if std::env::var("ZYNTAX_DISABLE_INTERP_OPTS").is_err() {
            let _stats = zyntax_compiler::run_interp_safe_opts(&mut owned);
        }
        zyntax_compiler::hir_dump::dump_module_to_dir(&owned, "post-opt-compile_module");

        // Check if we need to rebuild the backend for cross-module linking
        if self.backend.needs_rebuild_for_module(&owned) {
            log::debug!("[Runtime] Rebuilding JIT for cross-module symbol resolution");
            self.backend.rebuild_with_accumulated_symbols()?;
        }

        // Store function name -> ID mapping (resolve InternedString to actual string)
        // Also track which functions are async and store their signatures
        for (id, func) in &owned.functions {
            if let Some(name) = func.name.resolve_global() {
                self.function_ids.insert(name.clone(), *id);

                // Store the function signature for later use in call/call_async
                let native_sig = NativeSignature::from_hir_signature(&func.signature);
                self.function_signatures.insert(name.clone(), native_sig);

                // Track async functions by their original name
                // Async functions are transformed into {name}_new (constructor) and {name}_poll (poll)
                // We track the original name for call_async lookup
                //
                // Detection strategy:
                // 1. If func.signature.is_async, use the function name directly
                // 2. If function name ends with "_new", extract original name (async constructor)
                if func.signature.is_async {
                    self.async_functions.insert(name.clone());
                } else if name.ends_with("_new") {
                    // This is an async constructor - extract the original function name
                    let orig_name = name[..name.len() - 4].to_string();
                    self.async_functions.insert(orig_name);
                }
            }
        }

        // Filter Cranelift codegen to only functions transitively
        // reachable from `main`. Without this, Cranelift loops through
        // every prelude / stdlib / unused forward declaration in the
        // module, paying full per-function compile cost even on code
        // that never executes. Matches the filter installed at
        // `install_interp_jit_with` time so the two paths see the
        // same minimal set.
        let names = self.entry_names();
        let entry_names: Vec<&str> = names.iter().map(String::as_str).collect();
        let reachable = zyntax_compiler::reachable_function_ids(&owned, &entry_names);
        self.backend.set_only_compile_reachable(Some(reachable));

        // Compile the module
        self.backend.compile_module(&owned)?;

        // Finalize definitions to get function pointers
        self.backend.finalize_definitions()?;

        // Install the same module in the internal BC interpreter.
        // The interp owns execution dispatch — `call_function`
        // delegates here; beadie's `TieredAdapter` inside the interp
        // drives the single tier-up loop.
        if let Ok(mut interp) = self.interp.lock() {
            interp.compile_module(owned);
        }

        Ok(())
    }

    /// Compile source code using a language grammar
    ///
    /// This method parses the source code using the provided grammar, lowers the
    /// TypedAST to HIR, and compiles it into the runtime.
    ///
    /// # Arguments
    /// * `grammar` - The language grammar to use for parsing
    /// * `source` - The source code to compile
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};
    ///
    /// let grammar = LanguageGrammar::compile_zyn(include_str!("zig.zyn"))?;
    /// let mut runtime = ZyntaxRuntime::new()?;
    /// runtime.compile_with_grammar(&grammar, "fn main() -> i32 { 42 }")?;
    ///
    /// let result: i32 = runtime.call("main", &[])?;
    /// assert_eq!(result, 42);
    /// ```
    pub fn compile_with_grammar(
        &mut self,
        grammar: &crate::grammar::LanguageGrammar,
        source: &str,
    ) -> RuntimeResult<()> {
        // Parse source to TypedAST with plugin signatures for proper extern declarations
        let typed_program = grammar
            .parse_with_signatures(source, "<source>", &self.plugin_signatures)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Lower to HIR with grammar builtins
        let builtins: indexmap::IndexMap<String, String> = grammar
            .builtins()
            .functions
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut hir_module = self.lower_typed_program(typed_program, builtins)?;

        // Run the krio-driven async state-machine transform when the
        // `krio-async-backend` feature is on. No-op otherwise — the
        // legacy `compiler::async_support::AsyncCompiler` inside
        // `compile_module` continues to handle the lowering.
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        // Compile the module
        self.compile_module(&hir_module)
    }

    /// Lower a TypedProgram to HirModule.
    ///
    /// Runs the full lowering pipeline — struct/enum decl
    /// registration, import-trait/impl resolution, extern-decl
    /// registration, unresolved-type fixup, impl-block registration,
    /// abstract-trait synthesis, pattern engine, HIR lowering, async
    /// transform (when configured), monomorphization.
    ///
    /// Returns the lowered HIR ready for either the native JIT (via
    /// [`Self::compile_module`]) or the BC interpreter (via
    /// `crate::InterpRuntime::compile_module`).
    ///
    /// `builtins` is a name-keyed hint map for the SSA builder's
    /// `@builtin` extern resolution; pass an empty `IndexMap` when in
    /// doubt.
    pub fn lower_typed_program(
        &self,
        mut program: zyntax_typed_ast::TypedProgram,
        builtins: indexmap::IndexMap<String, String>,
    ) -> RuntimeResult<HirModule> {
        use zyntax_compiler::lowering::{LoweringConfig, LoweringContext};
        use zyntax_typed_ast::{
            type_registry::*, AstArena, InternedString, TypeRegistry, TypedDeclaration,
        };
        let fn_start = std::time::Instant::now();

        // Handler state (Phase 3): a `handler H for E { var s: T = init; ... }`
        // gets a synthesized `@reference` struct `H$state` holding its fields,
        // an `H$new()` constructor that allocates+initialises it, and an
        // implicit `self: H$state` prepended to every non-resumable op so the
        // body's `self.field` reads/writes go through the state region. This
        // MUST run before the registry is snapshotted into an immutable Arc
        // below — the later algebraic-effects pass can't register new types.
        synthesize_handler_state(&mut program);

        // Rebuild type registry from declarations (TypeRegistry is not serializable)
        // Scan for struct definitions (TypedDeclaration::Class) and register them
        // IMPORTANT: Only register types that don't already exist (abstract types are pre-registered by parser)
        for decl_node in &program.declarations {
            let decl_kind = match &decl_node.node {
                TypedDeclaration::Function(f) => {
                    format!("Function({})", f.name.resolve_global().unwrap_or_default())
                }
                TypedDeclaration::Class(c) => {
                    format!("Class({})", c.name.resolve_global().unwrap_or_default())
                }
                TypedDeclaration::Impl(_) => "Impl".to_string(),
                TypedDeclaration::Variable(_) => "Variable".to_string(),
                TypedDeclaration::Import(_) => "Import".to_string(),
                TypedDeclaration::Enum(_) => "Enum".to_string(),
                _ => "Other".to_string(),
            };
            if let TypedDeclaration::Class(class) = &decl_node.node {
                // Check if type is already registered (e.g., abstract types from parser)
                // If it exists but has 0 fields and the new one has fields, update it
                // (generic types like List<T> may be pre-registered as placeholders without fields)
                if let Some(existing) = program.type_registry.get_type_by_name(class.name) {
                    if existing.fields.is_empty() && !class.fields.is_empty() {
                        // Pre-registered placeholder with no fields - update with actual fields
                        let existing_id = existing.id;
                        let field_defs: Vec<FieldDef> = class
                            .fields
                            .iter()
                            .map(|f| FieldDef {
                                name: f.name,
                                ty: f.ty.clone(),
                                visibility: f.visibility,
                                mutability: f.mutability,
                                is_static: f.is_static,
                                span: f.span,
                                getter: None,
                                setter: None,
                                is_synthetic: false,
                            })
                            .collect();

                        let type_params: Vec<TypeParam> = class
                            .type_params
                            .iter()
                            .map(|tp| TypeParam {
                                name: tp.name,
                                bounds: vec![],
                                variance: Variance::Invariant,
                                default: tp.default.clone(),
                                span: tp.span,
                                is_const: tp.is_const,
                                const_ty: tp.const_ty.clone(),
                            })
                            .collect();

                        // Strict V1 reference-class lowering: propagate
                        // `@reference` annotation through the
                        // placeholder-update path so generic / pre-registered
                        // types (e.g. `List<T>`) still pick up reference
                        // semantics when annotated.
                        let is_reference = class.annotations.iter().any(|ann| {
                            ann.name
                                .resolve_global()
                                .as_deref()
                                .map(|n| n == "reference")
                                .unwrap_or(false)
                        });
                        let mut metadata: zyntax_typed_ast::type_registry::TypeMetadata =
                            Default::default();
                        metadata.is_reference = is_reference;

                        let type_def = TypeDefinition {
                            id: existing_id,
                            module: None,
                            name: class.name,
                            kind: TypeKind::Struct {
                                fields: field_defs.clone(),
                                is_tuple: false,
                            },
                            type_params,
                            constraints: vec![],
                            fields: field_defs,
                            methods: vec![],
                            constructors: vec![],
                            metadata,
                            span: class.span,
                        };
                        program.type_registry.register_type(type_def);
                    }
                    continue;
                }

                // Register the struct type. Use the TypeId from the declaration
                // node if available, otherwise generate a fresh one (Grammar2
                // parser sets ty: Type::Never instead of Type::Named).
                let type_id = if let zyntax_typed_ast::Type::Named { id, .. } = &decl_node.ty {
                    *id
                } else {
                    TypeId::next()
                };

                let field_defs: Vec<FieldDef> = class
                    .fields
                    .iter()
                    .map(|f| FieldDef {
                        name: f.name,
                        ty: f.ty.clone(),
                        visibility: f.visibility,
                        mutability: f.mutability,
                        is_static: f.is_static,
                        span: f.span,
                        getter: None,
                        setter: None,
                        is_synthetic: false,
                    })
                    .collect();

                let type_params: Vec<TypeParam> = class
                    .type_params
                    .iter()
                    .map(|tp| TypeParam {
                        name: tp.name,
                        bounds: vec![],
                        variance: Variance::Invariant,
                        default: tp.default.clone(),
                        span: tp.span,
                        is_const: tp.is_const,
                        const_ty: tp.const_ty.clone(),
                    })
                    .collect();

                // Strict V1 reference-class lowering: propagate `@reference`
                // annotation on fresh registration.
                let is_reference = class.annotations.iter().any(|ann| {
                    ann.name
                        .resolve_global()
                        .as_deref()
                        .map(|n| n == "reference")
                        .unwrap_or(false)
                });
                let mut metadata: zyntax_typed_ast::type_registry::TypeMetadata =
                    Default::default();
                metadata.is_reference = is_reference;

                let type_def = TypeDefinition {
                    id: type_id,
                    module: None,
                    name: class.name,
                    kind: TypeKind::Struct {
                        fields: field_defs.clone(),
                        is_tuple: false,
                    },
                    type_params,
                    constraints: vec![],
                    fields: field_defs,
                    methods: vec![],
                    constructors: vec![],
                    metadata,
                    span: class.span,
                };
                program.type_registry.register_type(type_def);
            }

            // Register enum types
            if let TypedDeclaration::Enum(enum_decl) = &decl_node.node {
                if program
                    .type_registry
                    .get_type_by_name(enum_decl.name)
                    .is_none()
                {
                    let type_id = if let zyntax_typed_ast::Type::Named { id, .. } = &decl_node.ty {
                        *id
                    } else {
                        TypeId::next()
                    };

                    let variants: Vec<zyntax_typed_ast::type_registry::VariantDef> = enum_decl
                        .variants
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            use zyntax_typed_ast::type_registry::VariantFields as VF;
                            use zyntax_typed_ast::typed_ast::TypedVariantFields as TVF;
                            let fields = match &v.fields {
                                TVF::Unit => VF::Unit,
                                TVF::Tuple(types) => VF::Tuple(types.clone()),
                                TVF::Named(fields) => VF::Named(
                                    fields
                                        .iter()
                                        .map(|f| FieldDef {
                                            name: f.name,
                                            ty: f.ty.clone(),
                                            visibility: Visibility::Public,
                                            mutability: f.mutability,
                                            is_static: false,
                                            span: f.span,
                                            getter: None,
                                            setter: None,
                                            is_synthetic: false,
                                        })
                                        .collect(),
                                ),
                            };
                            zyntax_typed_ast::type_registry::VariantDef {
                                name: v.name,
                                fields,
                                discriminant: Some(i as i64),
                                span: v.span,
                            }
                        })
                        .collect();

                    let type_params: Vec<TypeParam> = enum_decl
                        .type_params
                        .iter()
                        .map(|param| TypeParam {
                            name: param.name,
                            bounds: vec![],
                            variance: Variance::Invariant,
                            default: param.default.clone(),
                            span: param.span,
                            is_const: param.is_const,
                            const_ty: param.const_ty.clone(),
                        })
                        .collect();

                    let type_def = TypeDefinition {
                        id: type_id,
                        module: None,
                        name: enum_decl.name,
                        kind: TypeKind::Enum { variants },
                        type_params,
                        constraints: vec![],
                        fields: vec![],
                        methods: vec![],
                        constructors: vec![],
                        metadata: Default::default(),
                        span: enum_decl.span,
                    };
                    program.type_registry.register_type(type_def);
                }
            }
        }

        let arena = AstArena::new();
        // The module a program lowers under is the file it came from.
        // Naming every program `main` put them all in one module, which
        // is the wrong answer for anything that qualifies a name by the
        // module holding it.
        let module_name = program
            .source_files
            .first()
            .map(|file| crate::grammar::module_name_of(&file.name))
            .or_else(|| program.type_registry.current_module())
            .unwrap_or_else(|| InternedString::new_global("module"));
        let mut type_registry = program.type_registry.clone();

        // Process imports FIRST to load stdlib traits and impls
        // This merges declarations from imported modules into the program
        // and registers their opaque types in the type registry
        let t_imports = std::time::Instant::now();
        self.process_imports_for_traits(&mut program, &mut type_registry)?;
        let imports_ms = t_imports.elapsed().as_secs_f64() * 1000.0;

        // Now process extern declarations from the merged program (main + imports)
        // to ensure all opaque types are registered (needs &mut)
        let t_externs = std::time::Instant::now();
        self.process_extern_declarations_mut(&program, &mut type_registry)?;
        let externs_ms = t_externs.elapsed().as_secs_f64() * 1000.0;

        // IMPORTANT: Resolve all Type::Unresolved in the TypedAST before lowering
        // This mutates the program to replace Unresolved types with actual types from TypeRegistry
        // The compiler's type checker and SSA builder need resolved types
        let t_resolve = std::time::Instant::now();
        self.resolve_unresolved_types(&mut program, &type_registry);
        let resolve_ms = t_resolve.elapsed().as_secs_f64() * 1000.0;

        // IMPORTANT: Sync the program's type_registry with our local copy that has merged imports
        // This is needed because process_imports_for_traits merges into `type_registry` not `program.type_registry`
        program.type_registry = type_registry;

        // Register impl blocks before lowering
        zyntax_compiler::register_impl_blocks(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to register impl blocks: {:?}", e))
        })?;

        // Generate automatic trait implementations for abstract types
        zyntax_compiler::generate_abstract_trait_impls(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to generate abstract trait impls: {:?}", e))
        })?;

        // Register the generated impl blocks
        zyntax_compiler::register_impl_blocks(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to register generated impl blocks: {:?}", e))
        })?;

        // Wrap program's type registry in Arc for sharing (it now includes registered traits and impls)
        let type_registry_arc = std::sync::Arc::new(program.type_registry.clone());

        // Compiler-known built-in types route their static method
        // calls through the same mangled-symbol path stdlib
        // extern-struct methods do. For Fiber<T> the only
        // intercepted static method today is `Fiber.abort(err)` —
        // Wren-style abort from inside a fiber body — which maps to
        // the `krio_fiber_abort_with` runtime stub. Inject the
        // alias so the dot-static rewrite's
        // `resolve_associated_function_to_mangled` finds
        // `Fiber$abort` and emits a Call.
        let mut builtins = builtins;
        builtins
            .entry("Fiber$abort".to_string())
            .or_insert_with(|| "krio_fiber_abort_with".to_string());

        // Create LoweringConfig with builtins for extern call resolution
        let lowering_config = LoweringConfig {
            builtins,
            use_krio_async: cfg!(feature = "krio-async-backend"),
            // Where a program can begin, so lowering can skip the
            // bodies an import brought in that nothing reaches. The
            // same names the backend filters codegen against.
            entry_names: self.entry_names(),
            ..LoweringConfig::default()
        };

        // Phase timings for the compile-time work, behind an env var so
        // an ordinary run stays quiet. `lower_typed_program` is the
        // largest slice of a cold compile, and it is three phases with
        // very different costs, which a single number hides.
        let phase_trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
        let prologue_ms = fn_start.elapsed().as_secs_f64() * 1000.0;
        let engine_start = std::time::Instant::now();

        // Run pattern engine (term-rewriting passes on TypedAST)
        {
            let mut engine = pattern_engine::PatternEngine::new(pattern_engine::EngineConfig {
                target: pattern_engine::LoweringTarget::Cpu,
                max_iterations: 64,
                trace: cfg!(debug_assertions),
                verify_after: false,
            });
            engine.register_pass(normalization_pass::Pass);
            engine.register_pass(algebraic_effects_pass::Pass);
            engine.finalize().map_err(|e| {
                RuntimeError::Execution(format!("Pattern engine finalize error: {}", e))
            })?;
            let result = engine.run(&mut program, &type_registry_arc);
            if result.changed {
                log::debug!(
                    "[pattern_engine] {} rewrites fired in {} iterations",
                    result.rewrites_fired.len(),
                    result.iterations
                );
            }
        }

        let engine_elapsed = engine_start.elapsed();

        let mut lowering_ctx = LoweringContext::new(
            module_name,
            type_registry_arc.clone(),
            std::sync::Arc::new(std::sync::Mutex::new(arena)),
            lowering_config,
        );

        // Snapshot the runtime's wrapper-class registry (defaults +
        // any embedder-registered classes) into the lowering ctx so
        // every per-function SsaBuilder dispatches against the same
        // built-in set.
        lowering_ctx.set_builtin_registry(self.snapshot_builtin_registry());

        let lower_start = std::time::Instant::now();
        let mut hir_module = lowering_ctx
            .lower_program(&mut program)
            .map_err(|e| RuntimeError::Execution(format!("Lowering error: {:?}", e)))?;
        if phase_trace {
            let engine_ms = engine_elapsed.as_secs_f64() * 1000.0;
            let lower_ms = lower_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[LOWER-PHASES] prologue = {prologue_ms:.2} ms (imports {imports_ms:.2}, \
                 externs {externs_ms:.2}, resolve_types {resolve_ms:.2})  \
                 pattern_engine = {engine_ms:.2} ms  lower_program = {lower_ms:.2} ms"
            );
        }
        let epilogue_start = std::time::Instant::now();

        // Display lowering diagnostics (type inference warnings, etc.)
        lowering_ctx.display_diagnostics(&program);

        // Monomorphization
        let mono_start = std::time::Instant::now();
        zyntax_compiler::monomorphize_module(&mut hir_module)
            .map_err(|e| RuntimeError::Execution(format!("Monomorphization error: {:?}", e)))?;
        if phase_trace {
            eprintln!(
                "[LOWER-PHASES]   diagnostics = {:.2} ms  monomorphize = {:.2} ms  TOTAL = {:.2} ms",
                (mono_start - epilogue_start).as_secs_f64() * 1000.0,
                mono_start.elapsed().as_secs_f64() * 1000.0,
                fn_start.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Ok(hir_module)
    }

    /// Process import declarations to load stdlib traits and implementations.
    /// Thin wrapper over [`crate::import_chain::process_imports_for_traits`] —
    /// the actual logic is shared with `TieredRuntime`.
    fn process_imports_for_traits(
        &self,
        program: &mut zyntax_typed_ast::TypedProgram,
        type_registry: &mut zyntax_typed_ast::TypeRegistry,
    ) -> RuntimeResult<()> {
        crate::import_chain::process_imports_for_traits(
            &self.grammars,
            &self.plugin_signatures,
            &self.import_resolvers,
            &self.compiled_import_resolvers,
            &self.snapshot_modules,
            program,
            type_registry,
        )
    }

    /// Resolve all `Type::Unresolved` in the program. Wrapper over
    /// [`crate::import_chain::resolve_unresolved_types`].
    fn resolve_unresolved_types(
        &self,
        program: &mut zyntax_typed_ast::TypedProgram,
        type_registry: &zyntax_typed_ast::TypeRegistry,
    ) {
        crate::import_chain::resolve_unresolved_types(program, type_registry)
    }

    /// Process extern declarations to register opaque types in the
    /// `TypeRegistry`. Wrapper over
    /// [`crate::import_chain::process_extern_declarations_mut`].
    fn process_extern_declarations_mut(
        &self,
        program: &zyntax_typed_ast::TypedProgram,
        type_registry: &mut zyntax_typed_ast::TypeRegistry,
    ) -> RuntimeResult<()> {
        crate::import_chain::process_extern_declarations_mut(program, type_registry)
    }

    /// Register struct types from `TypedDeclaration::Class`. Wrapper over
    /// [`crate::import_chain::register_struct_declarations`].
    fn register_struct_declarations(
        &self,
        program: &zyntax_typed_ast::TypedProgram,
        type_registry: &mut zyntax_typed_ast::TypeRegistry,
    ) -> RuntimeResult<()> {
        crate::import_chain::register_struct_declarations(program, type_registry)
    }

    /// Get a function pointer by name
    pub fn get_function_ptr(&self, name: &str) -> Option<*const u8> {
        self.function_ids
            .get(name)
            .and_then(|id| self.backend.get_function_ptr(*id))
    }

    /// Call a function by name with the given arguments
    ///
    /// Arguments are automatically converted from `ZyntaxValue` and the result
    /// is converted back to the requested Rust type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result: i32 = runtime.call("add", &[10.into(), 20.into()])?;
    /// ```
    pub fn call<T: FromZyntax>(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<T> {
        let result = self.call_raw(name, args)?;
        T::from_zyntax(result).map_err(RuntimeError::from)
    }

    /// Calls a compiled `fiber def` constructor and returns an owning,
    /// thread-affine host handle.
    ///
    /// The returned handle is process-local runtime state. Applications must
    /// never serialize it; reconstruct a fresh fiber after runtime loss.
    pub fn call_fiber(
        &self,
        name: &str,
        args: &[ZyntaxValue],
    ) -> RuntimeResult<crate::ZyntaxFiber> {
        if !args.is_empty() {
            return Err(RuntimeError::ArgumentCount {
                expected: 0,
                got: args.len(),
            });
        }
        let entry = self
            .get_function_ptr(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;
        crate::ZyntaxFiber::from_entry(entry)
    }

    /// Call a function and get the raw ZyntaxValue result
    ///
    /// Note: This uses the stored function signature to determine the calling convention.
    /// For void functions, it uses a void-returning call. For native (i32, i64) functions,
    /// use `call_native` with a signature instead.
    pub fn call_raw(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<ZyntaxValue> {
        let ptr = self
            .get_function_ptr(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        // If we have a recorded HIR-derived signature, the function uses the
        // native scalar/pointer ABI (i64/f64/etc. returns), not the
        // DynamicValue ABI. Dispatch through `call_native_with_signature` so
        // scalar return values (e.g. `def main(): i64`) are decoded
        // correctly. Without this, an i64 return value gets reinterpreted as
        // a `DynamicValue { type_meta, value_ptr }` pair and the next
        // dereference SIGSEGVs — see `bench_fib.zynml` and friends.
        if let Some(sig) = self.function_signatures.get(name) {
            // SAFETY: We trust the function signature stored at load time
            // matches the JIT-compiled function's actual signature.
            return unsafe { call_native_with_signature(ptr, args, sig) };
        }

        // Convert arguments to DynamicValues
        let dynamic_args: Vec<DynamicValue> =
            args.iter().cloned().map(|v| v.into_dynamic()).collect();

        // Call the function using the variadic caller helper
        // SAFETY: We trust the caller has provided the correct function pointer
        // and matching arguments. from_dynamic is safe because call_dynamic_function
        // returns a valid DynamicValue.
        unsafe {
            let result = call_dynamic_function(ptr, &dynamic_args)?;
            ZyntaxValue::from_dynamic(result).map_err(RuntimeError::from)
        }
    }

    /// Call a JIT-compiled function with the specified signature
    ///
    /// This method dynamically constructs the function call based on the provided
    /// signature, converting ZyntaxValue arguments to the appropriate types.
    ///
    /// # Arguments
    /// * `name` - The function name
    /// * `args` - The arguments as ZyntaxValues
    /// * `signature` - The function signature describing parameter and return types
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zyntax_embed::{ZyntaxRuntime, NativeSignature, NativeType};
    ///
    /// // fn add(a: i32, b: i32) -> i32
    /// let sig = NativeSignature::new(&[NativeType::I32, NativeType::I32], NativeType::I32);
    /// let result = runtime.call_function("add", &[10.into(), 32.into()], &sig)?;
    /// assert_eq!(result, ZyntaxValue::Int(42));
    /// ```
    pub fn call_function(
        &self,
        name: &str,
        args: &[ZyntaxValue],
        signature: &NativeSignature,
    ) -> RuntimeResult<ZyntaxValue> {
        // Validate argument count.
        if args.len() != signature.params.len() {
            return Err(RuntimeError::Execution(format!(
                "Function '{}' expects {} arguments, got {}",
                name,
                signature.params.len(),
                args.len()
            )));
        }

        // Prefer the BC interpreter — it owns the single beadie
        // tier-up loop, so cold calls run bytecode and hot functions
        // tier up to Cranelift / LLVM. Fall back to the legacy native
        // Cranelift dispatch when the interpreter can't yet handle the
        // call (unsupported instruction such as an algebraic-effect
        // intrinsic, or the function isn't in the interp's HIR
        // module — e.g. registered as a foreign symbol).
        let interp_result = {
            let mut interp = self
                .interp
                .lock()
                .map_err(|e| RuntimeError::Execution(format!("interp lock poisoned: {e}")))?;
            interp.call_function(name, args.to_vec())
        };
        match interp_result {
            Ok(v) => Ok(v),
            Err(
                zyntax_compiler::hir_interp::InterpError::UnsupportedInstruction(_)
                | zyntax_compiler::hir_interp::InterpError::UnknownFunction(_),
            ) => {
                // Native dispatch via the underlying Cranelift JIT.
                let ptr = self
                    .get_function_ptr(name)
                    .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;
                // SAFETY: caller-supplied signature must match the
                // JIT-compiled function. Same contract as the original
                // `call_function` before the interpreter took over the
                // primary path.
                unsafe { call_native_with_signature(ptr, args, signature) }
            }
            Err(e) => Err(RuntimeError::Execution(format!(
                "interp dispatch failed: {e}"
            ))),
        }
    }

    // ── Execution-side forwarders (BC interp + beadie tier-up) ──
    //
    // These methods delegate to the internal interpreter state. The
    // user-facing API treats `ZyntaxRuntime` as the single runtime;
    // the interp is an implementation detail.

    /// Call a function directly with `ZyntaxValue` args, returning a
    /// `ZyntaxValue` result. No signature parameter — the interpreter
    /// dispatches by the function's HIR signature.
    pub fn call_function_raw(
        &self,
        name: &str,
        args: Vec<ZyntaxValue>,
    ) -> Result<ZyntaxValue, RuntimeError> {
        let mut interp = self
            .interp
            .lock()
            .map_err(|e| RuntimeError::Execution(format!("interp lock poisoned: {e}")))?;
        interp
            .call_function(name, args)
            .map_err(|e| RuntimeError::Execution(format!("interp dispatch failed: {e}")))
    }

    /// Register an extern "C" symbol callable from the BC interpreter
    /// (i64-funneled ABI). Mirrors `register_function` but on the
    /// interp side; use this when you're driving execution through
    /// the interp rather than the legacy JIT path.
    pub fn register_interp_symbol(&self, name: impl Into<String>, ptr: *const u8, param_count: u8) {
        if let Ok(mut interp) = self.interp.lock() {
            interp.register_symbol(name, ptr, param_count);
        }
    }

    /// Forward a slice of ZRTL plugin symbols (the shape
    /// `ZrtlPlugin::symbols_with_signatures()` returns) into the BC
    /// interpreter's FFI table.
    pub fn register_zrtl_symbols(&self, symbols: &[zyntax_compiler::zrtl::RuntimeSymbolInfo]) {
        if let Ok(mut interp) = self.interp.lock() {
            interp.register_zrtl_symbols(symbols);
        }
    }

    /// Register an additional built-in wrapper class. Called before
    /// any compilation runs; the registered class joins the
    /// compiler's defaults (`Fiber<T>`, future `SimdVector<T, N>`, ...)
    /// in the registry that `lower_typed_program` snapshots into
    /// the lowering ctx.
    ///
    /// Embedders use this to surface host-specific built-in types
    /// without modifying the compiler crate — same architectural
    /// seam ZRTL symbols use for runtime functions.
    pub fn register_builtin_class(
        &self,
        class: Arc<dyn zyntax_compiler::builtin_class::BuiltinClass + Send + Sync>,
    ) {
        if let Ok(mut reg) = self.builtin_registry.lock() {
            reg.register(class);
        }
    }

    /// Snapshot the current built-in registry into an
    /// `Arc<BuiltinRegistry>` for the lowering ctx. Each call
    /// produces a fresh registry that's a clone of the current
    /// classes — `lower_typed_program` calls this once per
    /// compilation so any classes registered AFTER the snapshot
    /// won't apply until the next compilation.
    fn snapshot_builtin_registry(&self) -> Arc<zyntax_compiler::builtin_class::BuiltinRegistry> {
        let mut snapshot = zyntax_compiler::builtin_class::BuiltinRegistry::new();
        if let Ok(reg) = self.builtin_registry.lock() {
            for class in reg.classes() {
                snapshot.register(class.clone());
            }
        }
        Arc::new(snapshot)
    }

    /// Install the BC interp → Cranelift opt [→ LLVM] tier ladder for
    /// hot-function promotion. Beadie's `TieredAdapter` (inside the
    /// interp) is the single tier-up orchestrator.
    /// The functions a host can enter a program through, as the
    /// registered grammars declare them.
    ///
    /// Zyntax has no entry point of its own. A language names one in
    /// its grammar metadata or a host configures it, and with nobody
    /// saying, nothing is treated as an entry.
    pub(super) fn entry_names(&self) -> Vec<String> {
        self.grammars
            .values()
            .filter_map(|grammar| grammar.entry_point().map(str::to_string))
            .collect()
    }

    pub fn install_interp_jit(&self) -> Result<(), CompilerError> {
        let names = self.entry_names();
        let mut interp = self
            .interp
            .lock()
            .map_err(|e| CompilerError::Backend(format!("interp lock poisoned: {e}")))?;
        interp.set_entry_names(names);
        interp.install_jit()
    }

    /// Install the tier ladder with a custom `TieredConfig`
    /// (promotion thresholds + tier-2 backend selection). See
    /// [`Self::install_interp_jit`].
    pub fn install_interp_jit_with(
        &self,
        config: zyntax_compiler::tiered_backend::TieredConfig,
    ) -> Result<(), CompilerError> {
        let names = self.entry_names();
        let mut interp = self
            .interp
            .lock()
            .map_err(|e| CompilerError::Backend(format!("interp lock poisoned: {e}")))?;
        interp.set_entry_names(names);
        interp.install_jit_with(config)
    }

    /// Diagnostic: profile counters for a function in the BC interp.
    pub fn interp_profile_for(&self, func_id: HirId) -> zyntax_compiler::hir_interp::ProfileSample {
        self.interp
            .lock()
            .map(|i| i.profile_for(func_id))
            .unwrap_or_default()
    }

    /// Diagnostic: every HirId that has a registered tier-up bound on
    /// the BC interp side. Used by tests to walk per-function state.
    pub fn interp_registered_function_ids(&self) -> Vec<HirId> {
        self.interp
            .lock()
            .map(|i| i.registered_function_ids().collect())
            .unwrap_or_default()
    }

    /// Diagnostic: snapshot of a function's beadie state — `Some` if
    /// the bead reports `compiled()`, else `None`.
    pub fn interp_function_compiled(&self, func_id: HirId) -> bool {
        self.interp
            .lock()
            .map(|i| i.bead_for(func_id).and_then(|b| b.compiled()).is_some())
            .unwrap_or(false)
    }

    /// Diagnostic: current beadie generation for a function — 0 means
    /// uncompiled (BC interp only), 1 means promoted to the first JIT
    /// tier (Cranelift opt), 2 means promoted to the second tier
    /// (LLVM, when the `llvm-backend` feature is on). Tests use this
    /// to assert tier-up actually crossed each rung.
    pub fn interp_function_generation(&self, func_id: HirId) -> u64 {
        self.interp
            .lock()
            .map(|i| i.bead_for(func_id).map(|b| b.generation()).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Call an async function, returning a Promise
    ///
    /// The promise can be awaited to get the result, or polled manually.
    ///
    /// For async functions compiled from Zyntax source, this automatically uses
    /// the state machine ABI:
    /// - `{fn}_new(params...) -> *mut StateMachine` - constructor
    /// - `{fn}_poll(state_machine, context) -> AsyncPollResult` - poll function
    ///
    /// # Example
    ///
    /// ```ignore
    /// let promise = runtime.call_async("fetch_data", &[url.into()])?;
    /// let result: String = promise.await_result()?;
    /// ```
    pub fn call_async(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<ZyntaxPromise> {
        // First, try the new Promise-based ABI:
        // The async function directly returns *Promise<T>
        if let Some(func_ptr) = self.get_function_ptr(name) {
            // Look up the stored signature for this function
            let signature = self
                .function_signatures
                .get(name)
                .cloned()
                .unwrap_or_else(|| {
                    // Fallback: infer signature from args (legacy behavior)
                    let params: Vec<NativeType> = args
                        .iter()
                        .map(|arg| match arg {
                            ZyntaxValue::Int(_) => NativeType::I64,
                            ZyntaxValue::Float(_) => NativeType::F64,
                            ZyntaxValue::Bool(_) => NativeType::Bool,
                            ZyntaxValue::String(_) => NativeType::Ptr,
                            ZyntaxValue::Null => NativeType::Ptr,
                            ZyntaxValue::Pointer(_) => NativeType::Ptr,
                            _ => NativeType::Ptr, // All other types (Array, Struct, Map, etc.)
                        })
                        .collect();
                    NativeSignature {
                        params,
                        ret: NativeType::Ptr,
                    }
                });

            let dynamic_args: Vec<DynamicValue> =
                args.iter().cloned().map(|v| v.into_dynamic()).collect();

            return Ok(unsafe {
                ZyntaxPromise::from_async_call(func_ptr, dynamic_args, &signature)
            });
        }

        // Fall back to legacy _new/_poll naming convention
        if self.async_functions.contains(name) {
            let constructor_name = format!("{}_new", name);
            let poll_name = format!("{}_poll", name);

            let init_ptr = self.get_function_ptr(&constructor_name).ok_or_else(|| {
                RuntimeError::FunctionNotFound(format!(
                    "Async constructor '{}' not found (for async function '{}')",
                    constructor_name, name
                ))
            })?;

            let poll_ptr = self.get_function_ptr(&poll_name).ok_or_else(|| {
                RuntimeError::FunctionNotFound(format!(
                    "Async poll function '{}' not found (for async function '{}')",
                    poll_name, name
                ))
            })?;

            let dynamic_args: Vec<DynamicValue> =
                args.iter().cloned().map(|v| v.into_dynamic()).collect();

            return Ok(ZyntaxPromise::with_poll_fn(
                init_ptr,
                poll_ptr,
                dynamic_args,
            ));
        }

        Err(RuntimeError::FunctionNotFound(format!(
            "Async function '{}' not found (tried both new Promise ABI and legacy _new/_poll)",
            name
        )))
    }

    /// Register an external function that can be called from Zyntax code
    pub fn register_function(&mut self, name: &str, ptr: *const u8, arg_count: usize) {
        self.external_functions.insert(
            name.to_string(),
            ExternalFunction {
                name: name.to_string(),
                ptr,
                arg_count,
            },
        );
        // Also register with the backend so Cranelift can resolve the symbol during JIT linking
        self.backend.register_runtime_symbol(name, ptr);
    }

    /// Register an external function together with a typed signature.
    ///
    /// Same as [`Self::register_function`] for the runtime symbol
    /// table, plus stores the signature on:
    ///
    /// - `plugin_signatures`, so `Grammar2::parse_with_signatures`
    ///   can use it for `@builtin` extern injection if the host opts
    ///   into that path.
    /// - The backend's `symbol_signatures` table, so call-site
    ///   lowering at codegen time uses the typed signature instead of
    ///   guessing `I64` returns and platform-default calling
    ///   conventions. Without this, statically-registered functions
    ///   collide with Zyntax-injected extern declarations on
    ///   `IncompatibleSignature` because the call site and the extern
    ///   decl describe the symbol differently.
    ///
    /// This is the static-registration equivalent of what
    /// [`Self::load_plugin`] does for `.zrtl` symbols. Hosts that
    /// statically link their builtins (e.g. an embedded UI DSL whose
    /// `$Foo$widget` functions live in the same binary) should prefer
    /// this over the un-typed [`Self::register_function`] so type
    /// inference and codegen agree on the symbol's shape.
    ///
    /// As with [`Self::register_function`], call
    /// [`Self::finalize_runtime_symbols`] after the last
    /// registration and before the first compile.
    pub fn register_function_typed(
        &mut self,
        name: &'static str,
        ptr: *const u8,
        sig: zyntax_compiler::zrtl::ZrtlSymbolSig,
    ) {
        self.external_functions.insert(
            name.to_string(),
            ExternalFunction {
                name: name.to_string(),
                ptr,
                arg_count: sig.param_count as usize,
            },
        );
        self.backend.register_runtime_symbol(name, ptr);

        // Mirror what `load_plugin` does for ZRTL symbol metadata.
        self.plugin_signatures.insert(name.to_string(), sig);

        // The backend's `symbol_signatures` table is what the
        // Cranelift call-site lowering reads when emitting calls to a
        // registered symbol — see
        // `crates/compiler/src/cranelift_backend.rs:2719`.
        // `register_symbol_signatures` takes
        // `&[RuntimeSymbolInfo]`, whose `name` field is `&'static str`
        // (copied to an owned `String` internally). Requiring
        // `&'static str` from the caller matches the underlying type
        // and avoids any lifetime extension trickery.
        let info = zyntax_compiler::zrtl::RuntimeSymbolInfo {
            name,
            ptr,
            sig: Some(sig),
        };
        self.backend.register_symbol_signatures(&[info]);
    }

    /// Rebuild the JIT module so symbols registered via
    /// [`Self::register_function`] become resolvable from
    /// subsequently-compiled modules.
    ///
    /// `register_function` records the symbol on the backend's
    /// accumulator, but the underlying Cranelift JITModule was
    /// constructed at [`Self::new`] time and only knows about the
    /// symbols that existed then. Plugin loaders
    /// ([`Self::load_plugin`], [`Self::load_plugins_from_directory`])
    /// call `rebuild_with_accumulated_symbols` internally to bridge
    /// this gap. Hosts that statically register their builtins
    /// (without going through `.zrtl` discovery) need an equivalent
    /// hook.
    ///
    /// Call once after batch-registering all builtins, before the
    /// first [`Self::compile_typed_program`] / [`Self::compile_module`] /
    /// [`Self::call`] invocation. Cheap; idempotent. Subsequent
    /// symbol registrations require another call.
    pub fn finalize_runtime_symbols(&mut self) -> RuntimeResult<()> {
        self.backend
            .rebuild_with_accumulated_symbols()
            .map_err(|e| RuntimeError::Execution(format!("rebuild_jit: {e}")))?;
        Ok(())
    }

    /// Hot-reload a function with new code
    pub fn hot_reload(
        &mut self,
        name: &str,
        function: &zyntax_compiler::HirFunction,
    ) -> RuntimeResult<()> {
        let id = self
            .function_ids
            .get(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        self.backend.hot_reload_function(*id, function)?;
        Ok(())
    }

    /// Get the compilation configuration
    pub fn config(&self) -> &CompilationConfig {
        &self.config
    }

    /// Get a mutable reference to the compilation configuration
    pub fn config_mut(&mut self) -> &mut CompilationConfig {
        &mut self.config
    }

    /// Load a ZRTL plugin from a file path
    ///
    /// This loads a native dynamic library (.zrtl, .so, .dylib, .dll) and
    /// registers all its exported symbols as external functions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// runtime.load_plugin("./my_runtime.zrtl")?;
    /// ```
    ///
    /// Requires the `dynamic-plugins` feature of `zyntax_compiler`
    /// (transitively enabled by zyntax_embed's default `native`
    /// feature). On wasm32 builds plugins must instead be registered
    /// statically via `register_static_plugin` (Phase C of the
    /// wasm-target plan).
    /// Register a statically-linked ZRTL plugin.
    ///
    /// Mirrors [`Self::load_plugin`] but takes a `zrtl::StaticPlugin`
    /// produced by the `zrtl_plugin!` macro's `static_plugin()` accessor
    /// instead of going through `dlopen`. Walks the plugin's symbol
    /// table (excluding the trailing null-name sentinel) and forwards
    /// each entry into:
    /// - the native backend's runtime-symbol table (via
    ///   [`Self::register_function`]),
    /// - `plugin_signatures` so `Grammar2::parse_with_signatures`
    ///   sees the same auto-boxing info that the dlopen path would,
    /// - the BC interpreter's FFI table so interpreter-mode dispatch
    ///   can call into the plugin too.
    ///
    /// After registration the JIT module is rebuilt (native only) so
    /// subsequent compiles can reach the new symbols.
    ///
    /// This is the wasm32 plugin entry point — there is no `dlopen` in
    /// a browser-hosted wasm module, so plugins are linked at build
    /// time and registered through this method. Native hosts use it to
    /// skip `dlopen` too.
    pub fn register_static_plugin(&mut self, plugin: zrtl::StaticPlugin) -> RuntimeResult<()> {
        self.register_static_plugin_deferred(plugin)?;
        self.backend.rebuild_with_accumulated_symbols()?;
        Ok(())
    }

    /// Register several statically-linked plugins, rebuilding the JIT
    /// module once for all of them rather than once apiece. The rebuild
    /// is the expensive half, so a host that knows its whole set up
    /// front pays for one.
    pub fn register_static_plugins(
        &mut self,
        plugins: impl IntoIterator<Item = zrtl::StaticPlugin>,
    ) -> RuntimeResult<()> {
        for plugin in plugins {
            self.register_static_plugin_deferred(plugin)?;
        }
        self.backend.rebuild_with_accumulated_symbols()?;
        Ok(())
    }

    /// Register one plugin's symbols, leaving the JIT rebuild to the
    /// caller so a batch can share a single one.
    fn register_static_plugin_deferred(&mut self, plugin: zrtl::StaticPlugin) -> RuntimeResult<()> {
        use std::ffi::CStr;
        use zyntax_compiler::zrtl::{
            RuntimeSymbolInfo, TypeTag, ZrtlSigFlags, ZrtlSymbolSig, MAX_PARAMS,
        };

        // Walk the SDK-side `ZrtlSymbol` array and build compiler-side
        // `RuntimeSymbolInfo` entries. Both sides are `#[repr(C)]` and
        // layout-compatible by ABI, but we rebuild through the safe
        // API rather than transmuting so the dependency boundary is
        // explicit.
        let mut runtime_symbols: Vec<RuntimeSymbolInfo> = Vec::new();
        for sym in plugin.symbols {
            // SAFETY: each `name` field in a `zrtl_plugin!`-generated
            // table is initialised from a `concat!("...", "\0")` static
            // literal — null-terminated and valid UTF-8. Skip on
            // unexpected non-UTF-8 rather than panic.
            let name: &'static str = unsafe {
                let cstr = CStr::from_ptr(sym.name);
                match cstr.to_str() {
                    // The pointer came from a `'static` literal in the
                    // plugin crate, so the returned `&str` lives for
                    // 'static as well.
                    Ok(s) => &*(s as *const str),
                    Err(_) => continue,
                }
            };

            // SAFETY: `sym.sig` is either null or points at a static
            // `ZrtlSymbolSig` in the plugin. The SDK and compiler
            // types are layout-compatible by `#[repr(C)]` design, so
            // we copy the fields through their `pub` u32 wrappers.
            let sig = if sym.sig.is_null() {
                None
            } else {
                let s = unsafe { &*sym.sig };
                let mut params = [TypeTag(0); MAX_PARAMS];
                for (i, p) in s.params.iter().enumerate().take(MAX_PARAMS) {
                    params[i] = TypeTag(p.0);
                }
                Some(ZrtlSymbolSig {
                    param_count: s.param_count,
                    flags: ZrtlSigFlags(s.flags.0),
                    return_type: TypeTag(s.return_type.0),
                    params,
                })
            };

            runtime_symbols.push(RuntimeSymbolInfo {
                name,
                ptr: sym.ptr,
                sig,
            });
        }

        // Mirror the dlopen path: register on the backend, stash
        // signatures, register signatures with the backend for
        // auto-boxing, then rebuild the JIT module so compiled code
        // can resolve the new symbols.
        for sym in &runtime_symbols {
            self.register_function(sym.name, sym.ptr, 0);
            if let Some(sig) = sym.sig {
                self.plugin_signatures.insert(sym.name.to_string(), sig);
            }
        }
        self.backend.register_symbol_signatures(&runtime_symbols);

        // Forward to the BC interpreter's FFI table so interp-mode
        // dispatch can reach these symbols as well.
        if let Ok(mut interp) = self.interp.lock() {
            interp.register_zrtl_symbols(&runtime_symbols);
        }

        Ok(())
    }

    #[cfg(feature = "dynamic-plugins")]
    pub fn load_plugin<P: AsRef<std::path::Path>>(&mut self, path: P) -> RuntimeResult<()> {
        use zyntax_compiler::zrtl::{ZrtlError, ZrtlPlugin};

        let plugin = ZrtlPlugin::load(path).map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Register all symbols from the plugin as runtime symbols
        // AND collect their signatures for type checking
        for symbol_info in plugin.symbols_with_signatures() {
            self.register_function(symbol_info.name, symbol_info.ptr, 0); // Arity unknown without type info

            // Store signature if available
            if let Some(sig) = symbol_info.sig {
                self.plugin_signatures
                    .insert(symbol_info.name.to_string(), sig);
            }
        }

        // Register symbol signatures for auto-boxing support in backend
        self.backend
            .register_symbol_signatures(plugin.symbols_with_signatures());

        // Rebuild the JIT module to include the new symbols
        self.backend.rebuild_with_accumulated_symbols()?;

        self.loaded_plugins.push(plugin);
        Ok(())
    }

    /// Load all ZRTL plugins from a directory
    ///
    /// Loads all `.zrtl` files from the specified directory.
    ///
    /// # Returns
    ///
    /// The number of plugins loaded successfully.
    pub fn load_plugins_from_directory<P: AsRef<std::path::Path>>(
        &mut self,
        dir: P,
    ) -> RuntimeResult<usize> {
        use zyntax_compiler::zrtl::ZrtlRegistry;

        let mut registry = ZrtlRegistry::new();
        let count = registry
            .load_directory(&dir)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Register all collected symbols AND their signatures
        for symbol_info in registry.collect_symbols_with_signatures() {
            self.register_function(symbol_info.name, symbol_info.ptr, 0);

            // Store signature if available
            if let Some(sig) = symbol_info.sig {
                self.plugin_signatures
                    .insert(symbol_info.name.to_string(), sig);
            }
        }

        // Register symbol signatures for auto-boxing support in backend
        let symbols_with_sigs = registry.collect_symbols_with_signatures();
        self.backend.register_symbol_signatures(&symbols_with_sigs);

        // Rebuild the JIT module to include all the new symbols
        self.backend.rebuild_with_accumulated_symbols()?;

        Ok(count)
    }

    /// Register an import resolver callback
    ///
    /// Import resolvers are called in order when resolving import statements.
    /// The first resolver to return `Ok(Some(source))` wins.
    ///
    /// # Example
    ///
    /// ```ignore
    /// runtime.add_import_resolver(Box::new(|path| {
    ///     if path == "my_module" {
    ///         Ok(Some("pub fn hello() -> i32 { 42 }".to_string()))
    ///     } else {
    ///         Ok(None) // Not found by this resolver
    ///     }
    /// }));
    /// ```
    pub fn add_import_resolver(&mut self, resolver: ImportResolverCallback) {
        self.import_resolvers.push(resolver);
    }

    /// Register a resolver for build-time parsed import artifacts.
    pub fn add_compiled_import_resolver(&mut self, resolver: CompiledImportResolverCallback) {
        self.compiled_import_resolvers.push(resolver);
    }

    /// Add a file-system based import resolver
    ///
    /// This resolver looks for modules in the specified directory using the given file extension.
    /// For import path "foo.bar" with extension "zig", it looks for:
    /// - `{base_path}/foo/bar.zig`
    /// - `{base_path}/foo.bar.zig` (dot-style path)
    ///
    /// # Arguments
    /// * `base_path` - The base directory to search for modules
    /// * `extension` - The file extension (without the dot), e.g., "zig", "hx", "py"
    ///
    /// # Example
    /// ```ignore
    /// // For Zig source files
    /// runtime.add_filesystem_resolver("./src", "zig");
    ///
    /// // For Haxe source files
    /// runtime.add_filesystem_resolver("./src", "hx");
    /// ```
    pub fn add_filesystem_resolver<P: AsRef<std::path::Path> + Send + Sync + 'static>(
        &mut self,
        base_path: P,
        extension: &str,
    ) {
        let base = base_path.as_ref().to_path_buf();
        let ext = extension.to_string();

        self.add_import_resolver(Box::new(move |module_path| {
            // Try slash-separated path (e.g., "foo.bar" -> "foo/bar.zig")
            let slash_path = module_path.replace('.', "/");
            let file_path = base.join(format!("{}.{}", slash_path, ext));
            if file_path.exists() {
                return std::fs::read_to_string(&file_path)
                    .map(Some)
                    .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e));
            }

            // Try dot-style path directly (e.g., "foo.bar" -> "foo.bar.zig")
            let dot_path = base.join(format!("{}.{}", module_path, ext));
            if dot_path.exists() {
                return std::fs::read_to_string(&dot_path)
                    .map(Some)
                    .map_err(|e| format!("Failed to read {}: {}", dot_path.display(), e));
            }

            Ok(None) // Not found
        }));
    }

    /// Resolve an import path using registered resolvers
    ///
    /// Returns the source code for the module if found.
    pub fn resolve_import(&self, module_path: &str) -> Result<Option<String>, String> {
        for resolver in &self.import_resolvers {
            match resolver(module_path) {
                Ok(Some(source)) => return Ok(Some(source)),
                Ok(None) => continue, // Try next resolver
                Err(e) => return Err(e),
            }
        }
        Ok(None) // Not found by any resolver
    }

    /// Get the number of registered import resolvers
    pub fn import_resolver_count(&self) -> usize {
        self.import_resolvers.len()
    }

    // ========================================================================
    // Multi-Language Grammar Registry
    // ========================================================================
    /// Install a language: its grammar, its standard library, and the
    /// type ids both were built against.
    ///
    /// This is the whole of what a language does to become usable in a
    /// runtime. Decoding the snapshot's modules reserves their
    /// build-time type ids, and doing it here, in the order the build
    /// wrote them, is why a language no longer has to remember to do
    /// it before parsing anything.
    ///
    /// Hold the snapshot for the process and install the same one into
    /// however many runtimes get built. Decoding happens once for the
    /// snapshot rather than once per runtime.
    ///
    /// Returns the grammar it registered, so a host that wants one of
    /// its own does not decode the same bytes twice.
    ///
    /// Plugins can load before or after this. A grammar's builtin
    /// names are recorded as the names they stand for and resolved
    /// when something asks, so nothing here depends on a plugin having
    /// arrived first.
    pub fn install_snapshot(
        &mut self,
        snapshot: Arc<crate::Snapshot>,
    ) -> RuntimeResult<LanguageGrammar> {
        let grammar =
            LanguageGrammar::from_compiled_bytes(snapshot.grammar_bytes()).map_err(|e| {
                RuntimeError::Execution(format!(
                    "snapshot for '{}' has an unreadable grammar: {e}",
                    snapshot.language()
                ))
            })?;
        let mut grammar = grammar;
        grammar.set_language(snapshot.language());
        self.register_grammar(snapshot.language(), grammar.clone());

        // Reserve the ids before anything can parse against them. The
        // build recorded what to reserve, so no module is decoded here
        // and one nobody imports is never decoded at all.
        snapshot.reserve_type_ids();

        for module in snapshot.module_names() {
            self.snapshot_modules.insert(
                (snapshot.language().to_string(), module.to_string()),
                Arc::clone(&snapshot),
            );
        }

        // A module that kept its source stays available to hosts that
        // would rather parse it than trust the artifact.
        let sources = snapshot;
        self.add_import_resolver(Box::new(move |module_name| {
            Ok(sources.module_source(module_name).map(str::to_string))
        }));

        Ok(grammar)
    }

    /// Register a language grammar with the runtime
    ///
    /// The language identifier is used to select the grammar when loading modules.
    /// File extensions from the grammar's metadata are automatically registered
    /// for extension-based language detection.
    ///
    /// # Arguments
    /// * `language` - The language identifier (e.g., "zig", "python", "haxe")
    /// * `grammar` - The compiled language grammar
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};
    ///
    /// let mut runtime = ZyntaxRuntime::new()?;
    /// runtime.register_grammar("zig", LanguageGrammar::compile_zyn_file("zig.zyn")?);
    /// runtime.register_grammar("python", LanguageGrammar::compile_zyn_file("python.zyn")?);
    ///
    /// // Now load modules by language
    /// runtime.load_module("zig", "pub fn add(a: i32, b: i32) i32 { return a + b; }")?;
    /// ```
    pub fn register_grammar(&mut self, language: &str, grammar: LanguageGrammar) {
        // Record what the grammar's builtin names stand for. Copying
        // the symbol's address here instead meant a plugin loaded
        // after its grammar never got an alias, since there was
        // nothing to copy at the time. The name is enough, and it is
        // true whenever the plugin arrives.
        for (alias, target) in &grammar.builtins().functions {
            self.builtin_aliases.insert(alias.clone(), target.clone());
        }

        let grammar = Arc::new(grammar);

        // Register file extensions from grammar metadata
        for ext in grammar.file_extensions() {
            let ext_key = if ext.starts_with('.') {
                ext.clone()
            } else {
                format!(".{}", ext)
            };
            self.extension_map.insert(ext_key, language.to_string());
        }

        self.grammars.insert(language.to_string(), grammar);
    }

    /// Register a grammar from a .zyn file
    ///
    /// Convenience method that compiles the grammar and registers it.
    ///
    /// # Example
    ///
    /// ```ignore
    /// runtime.register_grammar_file("zig", "grammars/zig.zyn")?;
    /// ```
    pub fn register_grammar_file<P: AsRef<Path>>(
        &mut self,
        language: &str,
        zyn_path: P,
    ) -> Result<(), GrammarError> {
        let grammar = LanguageGrammar::compile_zyn_file(zyn_path)?;
        self.register_grammar(language, grammar);
        Ok(())
    }

    /// Register a grammar from a pre-compiled .zpeg file
    ///
    /// # Example
    ///
    /// ```ignore
    /// runtime.register_grammar_zpeg("zig", "grammars/zig.zpeg")?;
    /// ```
    pub fn register_grammar_zpeg<P: AsRef<Path>>(
        &mut self,
        language: &str,
        zpeg_path: P,
    ) -> Result<(), GrammarError> {
        let grammar = LanguageGrammar::load(zpeg_path)?;
        self.register_grammar(language, grammar);
        Ok(())
    }

    /// Get a registered grammar by language name
    pub fn get_grammar(&self, language: &str) -> Option<&Arc<LanguageGrammar>> {
        self.grammars.get(language)
    }

    /// Get the language name for a file extension
    ///
    /// # Arguments
    /// * `extension` - The file extension (with or without leading dot)
    pub fn language_for_extension(&self, extension: &str) -> Option<&str> {
        let ext_key = if extension.starts_with('.') {
            extension.to_string()
        } else {
            format!(".{}", extension)
        };
        self.extension_map.get(&ext_key).map(|s| s.as_str())
    }

    /// List all registered language names
    pub fn languages(&self) -> Vec<&str> {
        self.grammars.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a language grammar is registered
    pub fn has_language(&self, language: &str) -> bool {
        self.grammars.contains_key(language)
    }

    /// Load a module from source code using a registered language grammar
    ///
    /// This parses the source code using the registered grammar for the language,
    /// lowers it to HIR, and compiles it into the runtime.
    ///
    /// Note: Functions are NOT automatically exported for cross-module linking.
    /// Use `load_module_with_exports` to specify which functions to export.
    ///
    /// # Arguments
    /// * `language` - The language identifier (must be previously registered)
    /// * `source` - The source code to compile
    ///
    /// # Returns
    /// The names of functions defined in the module
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zyntax_embed::{ZyntaxRuntime, LanguageGrammar};
    ///
    /// let mut runtime = ZyntaxRuntime::new()?;
    /// runtime.register_grammar("zig", LanguageGrammar::compile_zyn_file("zig.zyn")?);
    ///
    /// let functions = runtime.load_module("zig", r#"
    ///     pub fn add(a: i32, b: i32) i32 { return a + b; }
    ///     pub fn mul(a: i32, b: i32) i32 { return a * b; }
    /// "#)?;
    ///
    /// assert!(functions.contains(&"add".to_string()));
    /// let result: i32 = runtime.call("add", &[10.into(), 32.into()])?;
    /// ```
    pub fn load_module(&mut self, language: &str, source: &str) -> RuntimeResult<Vec<String>> {
        self.load_module_with_exports_and_filename(language, source, &[], None)
    }

    /// Load a module and export specified functions for cross-module linking
    ///
    /// Functions listed in `exports` will be made available as extern symbols
    /// for subsequent modules to call. A warning is printed if there's a name conflict.
    ///
    /// # Arguments
    /// * `language` - The language identifier (must be previously registered)
    /// * `source` - The source code to compile
    /// * `exports` - Names of functions to export for cross-module linking
    ///
    /// # Returns
    /// The names of functions defined in the module
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Module A exports 'add'
    /// runtime.load_module_with_exports("zig", r#"
    ///     pub fn add(a: i32, b: i32) i32 { return a + b; }
    /// "#, &["add"])?;
    ///
    /// // Module B can call 'add' via extern
    /// runtime.load_module("zig", r#"
    ///     extern fn add(a: i32, b: i32) i32;
    ///     pub fn double_add(a: i32, b: i32) i32 { return add(a, b) + add(a, b); }
    /// "#)?;
    /// ```
    pub fn load_module_with_exports(
        &mut self,
        language: &str,
        source: &str,
        exports: &[&str],
    ) -> RuntimeResult<Vec<String>> {
        self.load_module_with_exports_and_filename(language, source, exports, None)
    }

    /// Load a module with exports and a specific filename for diagnostics
    pub fn load_module_with_exports_and_filename(
        &mut self,
        language: &str,
        source: &str,
        exports: &[&str],
        filename: Option<&str>,
    ) -> RuntimeResult<Vec<String>> {
        let grammar = self.grammars.get(language).cloned().ok_or_else(|| {
            RuntimeError::Execution(format!(
                "Unknown language '{}'. Registered languages: {:?}",
                language,
                self.languages()
            ))
        })?;

        // Parse source to TypedAST with plugin signatures for proper extern declarations
        let mut typed_program = if let Some(fname) = filename {
            grammar
                .parse_with_signatures(source, fname, &self.plugin_signatures)
                .map_err(|e| RuntimeError::Execution(e.to_string()))?
        } else {
            grammar
                .parse_with_signatures(source, &format!("<{language}>"), &self.plugin_signatures)
                .map_err(|e| RuntimeError::Execution(e.to_string()))?
        };
        capture_runtime_events_from_program(
            &mut typed_program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );

        // Lower to HIR with grammar builtins
        let builtins: indexmap::IndexMap<String, String> = grammar
            .builtins()
            .functions
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut hir_module = self.lower_typed_program(typed_program, builtins)?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        // Collect function names before compilation
        // Use resolve_global() to get the actual string from InternedString
        let function_names: Vec<String> = hir_module
            .functions
            .values()
            .filter(|f| !f.is_external)
            .filter_map(|f| f.name.resolve_global())
            .collect();

        // Compile the module
        self.compile_module(&hir_module)?;

        // Export specified functions, remembering which language they
        // came from.
        for export_name in exports {
            self.export_function_from(export_name, Some(language))?;
        }

        Ok(function_names)
    }

    /// Export a compiled function for cross-module linking
    ///
    /// Makes the function available as an extern symbol for subsequent modules.
    /// Returns an error if the function doesn't exist or if there's a symbol conflict.
    ///
    /// # Arguments
    /// * `name` - The function name to export
    pub fn export_function(&mut self, name: &str) -> RuntimeResult<()> {
        self.export_function_from(name, None)
    }

    /// Export a function loaded as part of `language`'s module.
    ///
    /// Every language in a runtime shares one set of symbol names, so
    /// two languages cannot both export `add`. Taking the second one
    /// silently would leave the first language calling the second
    /// language's function.
    fn export_function_from(&mut self, name: &str, language: Option<&str>) -> RuntimeResult<()> {
        let holder = self.exported_by.get(name).map(String::as_str);
        if export_conflicts(holder, language) {
            let holding = holder.unwrap_or_default();
            let taking = language.unwrap_or_default();
            {
                return Err(RuntimeError::Execution(format!(
                    "'{name}' is exported by both '{holding}' and '{taking}'. Every language in a \
                     runtime shares one set of symbol names, so the second export would replace \
                     the first. Renaming it where it is imported does not help: an alias names \
                     the reference, not the symbol."
                )));
            }
        }
        if let Some(language) = language {
            self.exported_by
                .insert(name.to_string(), language.to_string());
        }

        // Get the function pointer from our function_ids map
        let ptr = self
            .get_function_ptr(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        // Check for conflict and warn
        if let Some(existing) = self.backend.check_export_conflict(name) {
            log::warn!(
                "Symbol conflict: '{}' is already exported at {:?}. Overwriting.",
                name,
                existing
            );
            // Use overwrite method to replace
            self.backend.export_function_ptr_overwrite(name, ptr);
        } else {
            // No conflict, use regular export
            self.backend
                .export_function_ptr(name, ptr)
                .map_err(|e| RuntimeError::Execution(e.to_string()))?;
        }

        if let Some(sig) = self.function_signatures.get(name) {
            if let Ok(mut interp) = self.interp.lock() {
                interp.register_symbol(name, ptr, sig.params.len() as u8);
            }
        }

        Ok(())
    }

    /// Check if exporting a function would cause a symbol conflict
    ///
    /// Returns Some with the existing pointer if a conflict exists.
    pub fn check_export_conflict(&self, name: &str) -> Option<*const u8> {
        self.backend.check_export_conflict(name)
    }

    /// Get all currently exported symbols
    pub fn exported_symbols(&self) -> Vec<(&str, *const u8)> {
        self.backend.exported_symbols()
    }

    /// Load a module from a file, auto-detecting the language from extension
    ///
    /// The file extension is used to look up the registered grammar.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Automatically uses "zig" grammar based on .zig extension
    /// runtime.load_module_file("./src/math.zig")?;
    /// ```
    pub fn load_module_file<P: AsRef<Path>>(&mut self, path: P) -> RuntimeResult<Vec<String>> {
        let path = path.as_ref();

        // Get the file extension
        let extension = path.extension().and_then(|e| e.to_str()).ok_or_else(|| {
            RuntimeError::Execution(format!("File '{}' has no extension", path.display()))
        })?;

        // Look up the language for this extension
        let language = self
            .language_for_extension(extension)
            .ok_or_else(|| {
                RuntimeError::Execution(format!(
                    "No grammar registered for extension '.{}'. Registered extensions: {:?}",
                    extension,
                    self.extension_map.keys().collect::<Vec<_>>()
                ))
            })?
            .to_string();

        // Read the source file
        let source = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::Execution(format!("Failed to read '{}': {}", path.display(), e))
        })?;

        // Use the file path as the filename for diagnostics
        let filename = path.to_string_lossy();
        self.load_module_with_exports_and_filename(&language, &source, &[], Some(&filename))
    }

    /// List all loaded function names
    pub fn functions(&self) -> Vec<&str> {
        self.function_ids.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a function is defined
    pub fn has_function(&self, name: &str) -> bool {
        self.function_ids.contains_key(name)
    }

    /// Get a reference to the plugin signatures
    ///
    /// This is useful for Grammar2 parsers that need to inject extern declarations
    /// for builtin functions with proper type signatures.
    pub fn plugin_signatures(&self) -> &HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig> {
        &self.plugin_signatures
    }

    /// Pointer for a registered external/plugin function, by symbol
    /// name. Returns `None` when the name isn't in the runtime symbol
    /// table — including the case where the function was compiled
    /// from source (use [`Self::get_function_ptr`] for those).
    ///
    /// Mainly useful for tests that want to assert a plugin's symbols
    /// were registered. Production code should call
    /// [`Self::call_function`] / [`Self::call_function_raw`] rather
    /// than dispatch through a raw pointer.
    pub fn external_function_ptr(&self, name: &str) -> Option<*const u8> {
        self.external_functions
            .get(name)
            .or_else(|| {
                // A grammar builtin stands for a symbol a plugin
                // provides, whenever that plugin turned up.
                self.builtin_aliases
                    .get(name)
                    .and_then(|target| self.external_functions.get(target))
            })
            .map(|f| f.ptr)
    }

    /// Register a runtime event sink callback.
    pub fn set_event_sink<F>(&mut self, sink: F)
    where
        F: Fn(&RuntimeEvent) + Send + Sync + 'static,
    {
        self.event_sink = Some(Arc::new(sink));
    }

    /// Clear the runtime event sink callback.
    pub fn clear_event_sink(&mut self) {
        self.event_sink = None;
    }

    /// View captured runtime events.
    pub fn runtime_events(&self) -> &[RuntimeEvent] {
        &self.runtime_events
    }

    /// Drain and return captured runtime events.
    pub fn drain_runtime_events(&mut self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut self.runtime_events)
    }

    /// Compile a TypedProgram directly (without parsing)
    ///
    /// This is useful when using Grammar2 to parse source code directly to TypedAST,
    /// bypassing the traditional grammar.parse() path.
    ///
    /// # Returns
    ///
    /// The names of functions defined in the module.
    pub fn compile_typed_program(
        &mut self,
        mut program: zyntax_typed_ast::TypedProgram,
    ) -> RuntimeResult<Vec<String>> {
        capture_runtime_events_from_program(
            &mut program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );
        // Lower to HIR, threading `config.builtins` (extern aliases) so
        // cooperative-async builtins like `sleep` →
        // `__zyntax_async_set_timeout` (and any caller-injected
        // aliases) resolve at SSA Call lowering. Prior to this we
        // passed an empty map here, which silently dropped the
        // runtime's config — so `rt.config_mut().builtins.insert(...)`
        // had no effect on `compile_typed_program`'s output.
        let builtins: indexmap::IndexMap<String, String> = self
            .config
            .builtins
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut hir_module = self.lower_typed_program(program, builtins)?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        // Collect function names before compilation
        let function_names: Vec<String> = hir_module
            .functions
            .values()
            .filter(|f| !f.is_external)
            .filter_map(|f| f.name.resolve_global())
            .collect();

        // Compile the module
        self.compile_module(&hir_module)?;

        Ok(function_names)
    }
}
