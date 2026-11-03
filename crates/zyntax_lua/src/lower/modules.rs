//! Modules for Lua lowering.

use super::*;

/// The files `require`d by name, transitively, each parsed once. A
/// name that is a standard library or has no file is left to
/// `require` at run time.
/// The files `require` names, loaded in turn, and whether every name
/// was found: one that was not is loaded while the program runs.
pub(super) fn load_required(first: &[String], main_file: &str) -> Result<(Vec<Loaded>, bool)> {
    let dir = std::path::Path::new(main_file)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let mut loaded: Vec<Loaded> = Vec::new();
    let mut queue: Vec<String> = first.to_vec();
    let mut seen: HashSet<String> = HashSet::new();
    let mut all_found = true;
    while let Some(name) = queue.pop() {
        if !seen.insert(name.clone())
            || crate::library::stdlib::LIBS.contains(&name.as_str())
            || name == "_G"
        {
            continue;
        }
        let relative = format!("{}.lua", name.replace('.', "/"));
        let path = dir.join(&relative);
        // A binary chunk is left to `require` at run time, which
        // reads it back.
        let Some(bytes) = std::fs::read(&path).ok().filter(|b| !crate::is_binary(b)) else {
            all_found = false;
            continue;
        };
        let text = crate::source_text(&bytes);
        let file = path.display().to_string();
        // Positioned in the file it is in, not the program's.
        let (ast, source) = crate::parse_chunk(&text, crate::LOAD_LEVEL).map_err(|e| {
            Error::Rejected(crate::parse::SyntaxError {
                line: None,
                message: e.one_line(&file, &text),
                offset: 0,
            })
        })?;
        let source = source.into_owned();
        let scopes = crate::scope::resolve(&ast);
        queue.extend(scopes.requires.iter().cloned());
        loaded.push(Loaded {
            name,
            file,
            source,
            ast,
            scopes,
        });
    }
    Ok((loaded, all_found))
}

/// Whether `#` on a table may be a `__len` metamethod's value, of
/// any type: a source of the program names the metamethod, or code
/// the program loads while it runs might.
pub(super) fn len_meta_possible(
    source: &str,
    scopes: &Scopes,
    loaded: &[Loaded],
    all_found: bool,
) -> bool {
    let names_it = |s: &str| s.contains("__len");
    !all_found
        || scopes.dynamic_code
        || names_it(source)
        || loaded
            .iter()
            .any(|m| m.scopes.dynamic_code || names_it(&m.source))
}

pub(super) fn line_starts_of(source: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(source.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

/// A module's variables and functions as declarations.
pub(super) fn declare(module: &Module<'_>, declarations: &mut Vec<TypedNode<TypedDeclaration>>) {
    for (name, ty) in module.module_vars.borrow().iter() {
        declarations.push(TypedNode::new(
            TypedDeclaration::Variable(TypedVariable {
                name: *name,
                ty: module.ir(*ty),
                mutability: Mutability::Mutable,
                initializer: None,
                visibility: Visibility::Public,
            }),
            Type::Unknown,
            Span::new(0, 0),
        ));
    }
    for f in module.functions.take() {
        let span = f.body.as_ref().map(|b| b.span).unwrap_or(Span::new(0, 0));
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(f),
            Type::Unknown,
            span,
        ));
    }
}

/// A chunk besides the main one (a required file, a loaded string) as
/// a function `lua$<tag>chunk($varargs) -> Any` returning its values,
/// with a variadic record code `lua$<tag>chunk$fn` a program calls it
/// through. Its declarations are appended; the record code's name is
/// returned.
/// `env_var` names the chunk's environment when it has one of its
/// own; `own_env` says the chunk starts it as the globals table,
/// rather than being handed one.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunk_module(
    scopes: &Scopes,
    inferred: &Inferred,
    ast: &ast::Ast,
    source: &str,
    file: &str,
    tag: &str,
    chunk_index: i64,
    env_var: Option<InternedString>,
    own_env: bool,
    library: &Library,
    slots: bool,
    declarations: &mut Vec<TypedNode<TypedDeclaration>>,
    hooks: &mut Vec<ShapeLayout>,
    registry: &mut zyntax_typed_ast::TypeRegistry,
    debug: DebugMode,
    debug_names: (&str, &str),
    stripped: bool,
) -> Result<(String, Option<St>)> {
    let mut module = Module {
        scopes,
        inferred,
        types: library.types.clone(),
        chunk: file,
        tag: tag.to_string(),
        chunk_index,
        env_var,
        line_starts: line_starts_of(source),
        stripped,
        fallible: crate::fallible::FALLIBLE.iter().copied().collect(),
        reentrant: crate::fallible::REENTRANT.iter().copied().collect(),
        raising: None,
        bounded: None,
        functions: RefCell::new(Vec::new()),
        module_vars: RefCell::new(Vec::new()),
        facts: RefCell::new(HashMap::new()),
        fresh_classes: RefCell::new(HashMap::new()),
        fresh_locals: RefCell::new(HashSet::new()),
        returns_first: RefCell::new(HashSet::new()),
        finders: RefCell::new(HashMap::new()),
        reentrant_helpers: RefCell::new(HashSet::new()),
        sorts: RefCell::new(HashMap::new()),
        debug: debug.info(),
        layouts: shape_layouts(inferred, chunk_index, slots),
        runs_once: false,
    };
    let span = Span::new(0, source.len());
    let lower = |module: &Module<'_>| -> Result<Vec<St>> {
        let mut main = Lowerer::new(module, CHUNK);
        main.returns = Returns::Dynamic;
        main.varargs = Some(intern("$varargs"));
        main.frames = module.debug.is_some();
        let mut statements = main.block(ast.nodes())?;
        if main.frames {
            statements.insert(0, main.debug_enter(CHUNK, span));
        }
        if main.entry_line {
            statements.insert(0, entry_line_save(span));
        }
        if let (Some(env), true) = (module.env_var, own_env) {
            statements.insert(
                0,
                assign(
                    var(env, Type::Any, span),
                    call("zl_globals_value", vec![], Type::Any, span),
                    span,
                ),
            );
        }
        if types::falls_through(ast.nodes()) {
            main.return_stmt(&[], span, &mut statements)?;
        }
        if main.frames {
            main.debug_function_records(CHUNK, 0, 0);
        }
        module.strip_line_restores(&mut statements);
        module.facts.borrow_mut().insert(
            CHUNK,
            RaiseFact {
                own: main.raised,
                checks_depth: false,
                dynamic: true,
                callees: main.raise_callees.clone(),
            },
        );
        Ok(statements)
    };
    lower(&module)?;
    let bounded = bounded_functions(&module.facts.borrow(), |f| {
        module.inferred.escaping.contains(&f) || module.scopes.func(f).escapes
    });
    let raising = raising_functions(&module.facts.borrow(), &bounded);
    module.bounded = Some(bounded);
    module.raising = Some(raising);
    module.functions.borrow_mut().clear();
    module.module_vars.borrow_mut().clear();
    module.facts.borrow_mut().clear();
    module.fresh_classes.borrow_mut().clear();
    module.fresh_locals.borrow_mut().clear();
    module.returns_first.borrow_mut().clear();
    module.finders.borrow_mut().clear();
    module.reentrant_helpers.borrow_mut().clear();
    module.sorts.borrow_mut().clear();
    if let Some(debug) = &module.debug {
        debug.records.borrow_mut().clear();
        debug.sites.set(0);
    }
    let statements = lower(&module)?;
    let chunk_name = format!("lua${tag}chunk");
    let code_name = format!("{chunk_name}$fn");
    module.functions.borrow_mut().push(typed_function(
        &chunk_name,
        vec![parameter(intern("$varargs"), module.anys(), span)],
        Type::Any,
        statements,
        span,
    ));
    // Its record code: a variadic taking the packed arguments.
    module.functions.borrow_mut().push(typed_function(
        &code_name,
        vec![
            parameter(intern("env"), module.anys(), span),
            parameter(intern("packed"), Type::Any, span),
        ],
        Type::Any,
        vec![ret(
            Some(call(
                &chunk_name,
                vec![call(
                    "zl_values",
                    vec![var(intern("packed"), Type::Any, span)],
                    module.anys(),
                    span,
                )],
                Type::Any,
                span,
            )),
            span,
        )],
        span,
    ));
    let register = module
        .debug
        .is_some()
        .then(|| debug_register(&module, debug_names, span));
    if let Some(env) = env_var {
        module.declare_module_var(env, Ty::Any);
    }
    declare(&module, declarations);
    module.declare_shapes(declarations, registry, hooks);
    Ok((code_name, register))
}
