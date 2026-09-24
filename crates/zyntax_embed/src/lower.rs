//! From a typed program to a HIR module, the way every runtime does it.
//!
//! The tiered runtime lowers programs it is handed, and a build script
//! lowers a standard library module for a snapshot. Both go through
//! here, so a module lowered ahead of time is the module the runtime
//! would have produced.

use std::collections::HashMap;
use std::sync::Arc;

use zyntax_compiler::builtin_class::BuiltinRegistry;
use zyntax_compiler::hir::HirModule;
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{AstArena, InternedString, TypedProgram};

use crate::grammar::LanguageGrammar;
use crate::import_chain::SnapshotModules;
use crate::runtime::{
    CompiledImportResolverCallback, ImportResolverCallback, RuntimeError, RuntimeResult,
};

/// What lowering reads from the runtime around it.
pub(crate) struct Inputs<'a> {
    pub grammars: &'a HashMap<String, Arc<LanguageGrammar>>,
    pub plugin_signatures: &'a HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    pub import_resolvers: &'a [ImportResolverCallback],
    pub compiled_import_resolvers: &'a [CompiledImportResolverCallback],
    pub snapshot_modules: &'a SnapshotModules,
    /// Extern aliases, name to runtime symbol.
    pub builtins: indexmap::IndexMap<String, String>,
    pub builtin_registry: Arc<BuiltinRegistry>,
    /// Names a program can be entered through; empty builds everything.
    pub entry_names: Vec<String>,
    /// Modules already lowered, beyond what the program's imports bring.
    pub prelowered: Vec<Arc<zyntax_compiler::bytecode::LazyModule>>,
    /// Functions and globals of lowered imports already installed where
    /// the program is going, which it links rather than brings.
    pub linked: Arc<std::collections::HashSet<zyntax_compiler::hir::HirId>>,
    /// An import naming its items brings those functions alone (see
    /// `import_chain::process_imports_inner`).
    pub selective: bool,
}

/// A lowered program.
pub(crate) struct Lowered {
    pub module: HirModule,
    /// The functions the program can be entered through, when it named
    /// an entry point; `None` when a host may call anything.
    pub entered: Option<Vec<String>>,
}

/// Lower a typed program: resolve its imports, register what it
/// declares, and build HIR for what it reaches.
pub(crate) fn lower_typed_program(
    mut program: TypedProgram,
    inputs: Inputs<'_>,
) -> RuntimeResult<Lowered> {
    use zyntax_typed_ast::TypedDeclaration;
    use zyntax_typed_ast::type_registry::*;

    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let mut at = web_time::Instant::now();
    let mut lap = |name: &str, at: &mut web_time::Instant| {
        if trace {
            eprintln!(
                "[LOWER] {name:<20} {:8.2} ms",
                at.elapsed().as_secs_f64() * 1000.0
            );
            *at = web_time::Instant::now();
        }
    };
    // Stateful handlers need their state struct, ctor and implicit
    // `self` synthesized before the registry snapshot.
    crate::runtime::synthesize_handler_state(&mut program);

    // Rebuild type registry from declarations
    for decl_node in &program.declarations {
        if let TypedDeclaration::Class(class) = &decl_node.node {
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

            let is_reference = class.annotations.iter().any(|ann| {
                ann.name
                    .resolve_global()
                    .as_deref()
                    .map(|n| n == "reference")
                    .unwrap_or(false)
            });
            let mut metadata: TypeMetadata = Default::default();
            metadata.is_reference = is_reference;

            let type_def = TypeDefinition {
                id: type_id,
                module: None,
                name: class.name,
                kind: TypeKind::Struct {
                    fields: field_defs.clone(),
                    is_tuple: false,
                },
                type_params: vec![],
                constraints: vec![],
                fields: field_defs,
                methods: vec![],
                constructors: vec![],
                metadata,
                span: class.span,
            };
            program.type_registry.register_type(type_def);
        }
    }

    let mut type_registry = program.type_registry.clone();
    lap("registry", &mut at);

    // Imports first, so what they declare is in the program before
    // anything resolves against it. Modules that arrive already lowered
    // are collected for the lowering to link against.
    let mut prelowered = inputs.prelowered;
    crate::import_chain::process_imports_for_traits(
        inputs.grammars,
        inputs.plugin_signatures,
        inputs.import_resolvers,
        inputs.compiled_import_resolvers,
        inputs.snapshot_modules,
        &mut program,
        &mut type_registry,
        &mut prelowered,
        inputs.selective,
    )?;

    lap("imports", &mut at);

    crate::import_chain::process_extern_declarations_mut(&program, &mut type_registry)?;
    crate::import_chain::resolve_unresolved_types(&mut program, &type_registry);
    program.type_registry = type_registry;
    lap("externs+resolve", &mut at);

    zyntax_compiler::register_impl_blocks(&mut program)
        .map_err(|e| RuntimeError::Execution(format!("Failed to register impl blocks: {:?}", e)))?;
    zyntax_compiler::generate_abstract_trait_impls(&mut program).map_err(|e| {
        RuntimeError::Execution(format!("Failed to generate abstract trait impls: {:?}", e))
    })?;
    zyntax_compiler::register_impl_blocks(&mut program).map_err(|e| {
        RuntimeError::Execution(format!("Failed to register generated impl blocks: {:?}", e))
    })?;
    lap("impl blocks", &mut at);

    let arena = AstArena::new();
    // The module a program lowers under is the file it came from.
    let module_name = program
        .source_files
        .first()
        .map(|file| crate::grammar::module_name_of(&file.name))
        .or_else(|| program.type_registry.current_module())
        .unwrap_or_else(|| InternedString::new_global("module"));
    let type_registry = Arc::new(program.type_registry.clone());

    // `Fiber.abort(err)` inside a fiber body reaches the runtime's abort.
    let mut builtins = inputs.builtins;
    builtins
        .entry("Fiber$abort".to_string())
        .or_insert_with(|| "krio_fiber_abort_with".to_string());

    let lowering_config = LoweringConfig {
        builtins,
        use_krio_async: cfg!(feature = "krio-async-backend"),
        entry_names: inputs.entry_names,
        prelowered,
        linked: inputs.linked,
        ..LoweringConfig::default()
    };

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
        let _result = engine.run(&mut program, &type_registry);
    }
    lap("pattern engine", &mut at);

    let mut lowering_ctx = LoweringContext::new(
        module_name,
        type_registry,
        Arc::new(std::sync::Mutex::new(arena)),
        lowering_config,
    );
    lowering_ctx.set_builtin_registry(inputs.builtin_registry);

    let mut module = lowering_ctx
        .lower_program(&mut program)
        .map_err(|e| RuntimeError::Execution(format!("Lowering error: {:?}", e)))?;

    lowering_ctx.display_diagnostics(&program);
    lap("lower_program", &mut at);

    zyntax_compiler::monomorphize_module(&mut module)
        .map_err(|e| RuntimeError::Execution(format!("Monomorphization error: {:?}", e)))?;
    lap("monomorphize", &mut at);

    Ok(Lowered {
        module,
        entered: lowering_ctx.entered_functions(),
    })
}
