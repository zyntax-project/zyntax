//! Loaded for Lua lowering.

use super::*;

/// A chunk `load` compiles while the program runs, as a program of its
/// own: the chunk's function, and `lua$l<k>$init(env) -> Any`, which
/// registers the chunk's name for positions, takes the globals table
/// the chunk reads (the program's when `env` is nil) and returns the
/// chunk as a function value.
pub(crate) fn loaded_program(
    ast: &ast::Ast,
    source: &str,
    chunk_name: &str,
    index: i64,
    stripped: bool,
    library: &Library,
) -> Result<TypedProgram> {
    let (scopes, inferred) = loaded_types(ast);
    let references = NamedReferences::start();
    let lowered = loaded_declarations(
        &scopes, &inferred, ast, source, chunk_name, index, stripped, library,
    );
    let named = references.finish();
    let (mut declarations, registry) = lowered?;
    // The library's functions the chunk names, and no others: the rest
    // are already where the chunk will run.
    let mut import = library_import();
    if let TypedDeclaration::Import(i) = &mut import.node {
        i.items = named
            .into_iter()
            .map(|name| zyntax_typed_ast::typed_ast::TypedImportItem::Named { name, alias: None })
            .collect();
    }
    declarations.push(import);
    let span = Span::new(0, source.len());
    Ok(TypedProgram {
        checker_interface: None,
        declarations,
        language: Some(intern("lua")),
        span,
        source_files: vec![zyntax_typed_ast::source::SourceFile::new(
            chunk_name.to_string(),
            source.to_string(),
        )],
        type_registry: registry,
    })
}

/// The scopes and types of a chunk `load` compiles.
pub(crate) fn loaded_types(ast: &ast::Ast) -> (Scopes, Inferred) {
    let program = DebugMode::program();
    let mut scopes = crate::scope::resolve_loaded(ast, program.rebinds);
    scopes.dynamic_globals = true;
    scopes.len_meta = true;
    if !program.on && !scopes.debug {
        scopes.prune_unused_functions();
    }
    // The chunk is a function value, called by whoever `load` gave it to.
    scopes.funcs[CHUNK.0 as usize].escapes = true;
    let inferred = types::infer(&scopes, ast);
    (scopes, inferred)
}

/// The declarations of a loaded chunk: its functions and its `init`.
#[allow(clippy::too_many_arguments)]
pub(super) fn loaded_declarations(
    scopes: &Scopes,
    inferred: &Inferred,
    ast: &ast::Ast,
    source: &str,
    chunk_name: &str,
    index: i64,
    stripped: bool,
    library: &Library,
) -> Result<(
    Vec<TypedNode<TypedDeclaration>>,
    zyntax_typed_ast::TypeRegistry,
)> {
    let tag = format!("l{index}$");
    let env_var = intern(&format!("lua$l{index}$env"));
    let mut declarations = Vec::new();
    // Its tables are plain: the program's hooks know nothing of a
    // chunk's shapes, and it defines none of its own.
    let mut registry = library.type_registry.clone();
    let mut hooks = Vec::new();
    // A chunk of a program that keeps its call stack keeps it too, and
    // a chunk that reaches the debug library has the chunks loaded
    // after it keep what it may ask of their functions.
    let debug = DebugMode::of(scopes).join(DebugMode::program());
    debug.remember();
    let (code_name, register) = chunk_module(
        scopes,
        inferred,
        ast,
        source,
        chunk_name,
        &tag,
        index,
        Some(env_var),
        false,
        library,
        false,
        &mut declarations,
        &mut hooks,
        &mut registry,
        debug,
        (chunk_name, chunk_name),
        stripped,
    )?;
    let span = Span::new(0, source.len());
    let env = intern("env");
    let mut statements = vec![expr_stmt(call(
        "zl_chunk_add",
        vec![
            string_literal(crate::source_bytes(chunk_name).into_owned(), span),
            int_lit(index, span),
        ],
        prim(PrimitiveType::Unit),
        span,
    ))];
    statements.extend(register);
    statements.push(assign(
        var(env_var, Type::Any, span),
        var(env, Type::Any, span),
        span,
    ));
    statements.extend(chunk_value(
        &code_name,
        index,
        debug.on,
        library.types.anys(),
        span,
    ));
    let init = typed_function(
        &format!("lua${tag}init"),
        vec![parameter(env, Type::Any, span)],
        Type::Any,
        statements,
        span,
    );
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(init),
        Type::Unknown,
        span,
    ));
    Ok((declarations, registry))
}

/// A program of one function, `lua$error_text(err) -> Any`, entering
/// the library's text for an uncaught table error. Compiled by the host
/// only when such an error is reported, so no chunk reaches it.
pub(crate) fn error_text_program(library: &Library) -> TypedProgram {
    let span = Span::new(0, 0);
    let err = intern("err");
    let entry = typed_function(
        ERROR_TEXT_ENTRY,
        vec![parameter(err, Type::Any, span)],
        Type::Any,
        vec![ret(
            Some(call(
                library::ERROR_TEXT,
                vec![var(err, Type::Any, span)],
                Type::Any,
                span,
            )),
            span,
        )],
        span,
    );
    TypedProgram {
        checker_interface: None,
        declarations: vec![
            TypedNode::new(TypedDeclaration::Function(entry), Type::Unknown, span),
            library_import(),
        ],
        language: Some(intern("lua")),
        span,
        source_files: Vec::new(),
        type_registry: library.type_registry.clone(),
    }
}
