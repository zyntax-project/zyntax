//! Program for Lua lowering.

use super::*;

/// The whole chunk as a program.
pub(crate) fn program(
    ast: &ast::Ast,
    source: &str,
    file: &str,
    library: &Library,
    entry_kind: Entry,
) -> Result<TypedProgram> {
    let started = std::time::Instant::now();
    let mut scopes = crate::scope::resolve(ast);
    let (mut loaded, all_found) = load_required(&scopes.requires, file)?;
    // Files share their globals through the table, as does code run
    // at run time, which may write any global the program reads.
    let shared = entry_kind == Entry::Host
        || !loaded.is_empty()
        || !all_found
        || scopes.dynamic_code
        || loaded.iter().any(|m| m.scopes.dynamic_code);
    if shared {
        scopes.dynamic_globals = true;
        for m in &mut loaded {
            m.scopes.dynamic_globals = true;
        }
    }
    // A table one file hands another may get its metatable there, where
    // the first file's types do not see it.
    let sets_metatables = |s: &Scopes| {
        s.unseen_metatables || s.globals.contains("setmetatable") || s.globals.contains("debug")
    };
    if (!loaded.is_empty() || !all_found)
        && (!all_found
            || sets_metatables(&scopes)
            || loaded.iter().any(|m| sets_metatables(&m.scopes)))
    {
        scopes.unseen_metatables = true;
        for m in &mut loaded {
            m.scopes.unseen_metatables = true;
        }
    }
    let len_meta = len_meta_possible(source, &scopes, &loaded, all_found);
    scopes.len_meta = len_meta;
    for m in &mut loaded {
        m.scopes.len_meta = len_meta;
    }
    if !scopes.debug && loaded.iter().all(|m| !m.scopes.debug) {
        scopes.prune_unused_functions();
        for m in &mut loaded {
            m.scopes.prune_unused_functions();
        }
    }
    crate::trace_phase("scopes", started);
    let started = std::time::Instant::now();
    let inferred = types::infer(&scopes, ast);
    let module_inferred: Vec<Inferred> = loaded
        .iter()
        .map(|m| types::infer(&m.scopes, &m.ast))
        .collect();
    crate::trace_phase("infer", started);
    let started = std::time::Instant::now();
    // Every file of a program that reaches the debug library keeps the
    // call stack, and so does every chunk it loads.
    let debug = loaded.iter().fold(DebugMode::of(&scopes), |d, m| {
        d.join(DebugMode::of(&m.scopes))
    });
    debug.remember();
    let line_starts = line_starts_of(source);
    // A program that assigns `_ENV` has an environment of its own,
    // the globals table until then.
    let env_var = scopes
        .global_writes
        .contains_key("_ENV")
        .then(|| intern(MAIN_ENV));
    let mut module = Module {
        scopes: &scopes,
        inferred: &inferred,
        types: library.types.clone(),
        chunk: file,
        tag: String::new(),
        chunk_index: 0,
        env_var,
        line_starts,
        stripped: false,
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
        layouts: shape_layouts(&inferred, 0, true),
        runs_once: entry_kind == Entry::Program,
    };
    let span = Span::new(0, source.len());
    // Lowered twice: the first time finds which functions may raise, so
    // the second checks after calls to those alone.
    let lower_chunk = |module: &Module<'_>| -> Result<Vec<St>> {
        let mut main = Lowerer::new(module, CHUNK);
        main.frames = module.debug.is_some();
        main.returns = Returns::Fixed(Vec::new());
        // `...` at the main chunk: the script's arguments, in a module
        // variable since a segment is a function of its own.
        main.varargs = Some(intern(MAIN_VARARGS));
        let mut statements = if scopes.split_chunk {
            // Each segment is a function; the chunk calls them in turn
            // and stops at an error or a `return`.
            let segments = main.segments(ast.nodes())?;
            let unit = prim(PrimitiveType::Unit);
            let mut driver = Vec::new();
            let returned = module
                .module_vars
                .borrow()
                .iter()
                .any(|(n, _)| *n == intern(RETURNED));
            for (k, mut body) in segments.into_iter().enumerate() {
                if main.entry_line {
                    body.insert(0, entry_line_save(span));
                }
                body.push(ret(None, span));
                module.strip_line_restores(&mut body);
                let name = format!("{CHUNK_FN}${k}");
                module.functions.borrow_mut().push(typed_function(
                    &name,
                    Vec::new(),
                    unit.clone(),
                    body,
                    span,
                ));
                driver.push(expr_stmt(call(&name, vec![], unit.clone(), span)));
                driver.push(if_(
                    binary(
                        BinaryOp::Ne,
                        Lowerer::pending(span),
                        nil(span),
                        prim(PrimitiveType::Bool),
                        span,
                    ),
                    vec![ret(None, span)],
                    None,
                    span,
                ));
                if returned {
                    driver.push(if_(
                        var(intern(RETURNED), prim(PrimitiveType::Bool), span),
                        vec![ret(None, span)],
                        None,
                        span,
                    ));
                }
            }
            driver.push(ret(None, span));
            driver
        } else {
            let mut statements = main.block(ast.nodes())?;
            if main.entry_line {
                statements.insert(0, entry_line_save(span));
            }
            if types::falls_through(ast.nodes()) {
                statements.push(ret(None, span));
            }
            module.strip_line_restores(&mut statements);
            statements
        };
        let presets = main.preset_globals(span);
        statements.splice(0..0, presets);
        if main.frames {
            statements.insert(0, main.debug_enter(CHUNK, span));
            main.debug_function_records(CHUNK, 0, 0);
        }
        if let Some(env) = module.env_var {
            statements.insert(
                0,
                assign(
                    var(env, Type::Any, span),
                    call("zl_globals_value", vec![], Type::Any, span),
                    span,
                ),
            );
        }
        statements.insert(
            0,
            assign(
                var(intern(MAIN_VARARGS), module.anys(), span),
                call("zl_script_args", vec![], module.anys(), span),
                span,
            ),
        );
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
    let references = NamedReferences::start();
    lower_chunk(&module)?;
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
    crate::trace_phase("lower 1", started);
    let started = std::time::Instant::now();
    let statements = lower_chunk(&module)?;
    crate::trace_phase("lower 2", started);
    let chunk_fn = typed_function(
        CHUNK_FN,
        Vec::new(),
        prim(PrimitiveType::Unit),
        statements,
        span,
    );
    if let Some(env) = module.env_var {
        module.declare_module_var(env, Ty::Any);
    }
    let register = debug.on.then(|| {
        let source_name = format!("@{file}");
        debug_register(&module, (&source_name, &chunk_id(module.chunk)), span)
    });
    let mut declarations = Vec::new();
    declare(&module, &mut declarations);
    let mut registry = library.type_registry.clone();
    let mut hooks = Vec::new();
    module.declare_shapes(&mut declarations, &mut registry, &mut hooks);
    declarations.push(TypedNode::new(
        TypedDeclaration::Variable(TypedVariable {
            name: intern(MAIN_VARARGS),
            ty: module.anys(),
            mutability: Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        Span::new(0, 0),
    ));

    // Each required file is a chunk of its own, a function the program
    // enters through `package.preload`, and named in positions by its
    // number.
    let mut preloads: Vec<St> = Vec::new();
    preloads.extend(register);
    for (k, m) in loaded.iter().enumerate() {
        let tag = format!("m${}$", m.name.replace('.', "$"));
        let env_var = m
            .scopes
            .global_writes
            .contains_key("_ENV")
            .then(|| intern(&format!("lua${tag}env")));
        // Positions name the file as `require` found it.
        let found_as = format!("./{}.lua", m.name.replace('.', "/"));
        let source_name = format!("@{found_as}");
        crate::dump::note_chunk(k as i64 + 1, &m.source, &source_name, false);
        let (code_name, register) = chunk_module(
            &m.scopes,
            &module_inferred[k],
            &m.ast,
            &m.source,
            &m.file,
            &tag,
            k as i64 + 1,
            env_var,
            true,
            library,
            true,
            &mut declarations,
            &mut hooks,
            &mut registry,
            debug,
            (&source_name, &chunk_id(&found_as)),
            false,
        )?;
        preloads.push(expr_stmt(call(
            "zl_chunk_add",
            vec![
                str_lit(&chunk_id(&found_as), span),
                int_lit(k as i64 + 1, span),
            ],
            prim(PrimitiveType::Unit),
            span,
        )));
        preloads.extend(register);
        // The chunk as a function value, under its key.
        let key = (k as i64 + 1) << 32;
        let record = call(
            "zb_func_new",
            vec![
                code_of(&code_name, span),
                int_lit(VARIADIC_ARITY, span),
                node(
                    TypedExpression::Array(vec![call(
                        "zb_box_i64",
                        vec![int_lit(key, span)],
                        Type::Any,
                        span,
                    )]),
                    module.anys(),
                    span,
                ),
            ],
            Type::Any,
            span,
        );
        let chunk_var = intern(&format!("$chunk{k}"));
        preloads.push(let_(chunk_var, Type::Any, record, span));
        if debug.on {
            preloads.push(expr_stmt(call(
                "zl_dbg_closure",
                vec![int_lit(key, span), var(chunk_var, Type::Any, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        preloads.push(expr_stmt(call(
            "zl_preload_module",
            vec![str_lit(&m.name, span), var(chunk_var, Type::Any, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
    }

    let hooks = shape_hooks(&module, &hooks, span);
    let mut entry_body = Vec::new();
    if !hooks.is_empty() {
        entry_body.push(expr_stmt(call(
            library::SHAPE_HOOKS_INSTALL,
            library::SHAPE_HOOKS
                .iter()
                .map(|hook| code_of(&shape_hook_name(hook), span))
                .collect(),
            prim(PrimitiveType::Unit),
            span,
        )));
    }
    entry_body.extend([
        assign(
            var(intern(library::CHUNK), prim(PrimitiveType::String), span),
            str_lit(&chunk_id(module.chunk), span),
            span,
        ),
        if scopes.dynamic_globals {
            stmt(
                TypedStatement::Block(TypedBlock {
                    statements: vec![
                        assign(
                            var(intern(library::GLOBALS), module.ir(Ty::Table), span),
                            call("zl_globals_table", vec![], module.ir(Ty::Table), span),
                            span,
                        ),
                        assign(
                            var(
                                intern(library::capi::SHARED),
                                prim(PrimitiveType::I64),
                                span,
                            ),
                            int_lit(1, span),
                            span,
                        ),
                    ],
                    span,
                }),
                span,
            )
        } else {
            stmt(
                TypedStatement::Block(TypedBlock {
                    statements: Vec::new(),
                    span,
                }),
                span,
            )
        },
    ]);
    entry_body.extend(preloads);
    entry_body.push(expr_stmt(call("zl_arr_shared", vec![], Type::Any, span)));
    if entry_kind == Entry::Program {
        entry_body.extend([
            expr_stmt(call(CHUNK_FN, vec![], prim(PrimitiveType::Unit), span)),
            expr_stmt(call(
                "zl_report_pending",
                vec![],
                prim(PrimitiveType::Unit),
                span,
            )),
        ]);
    }
    entry_body.push(ret(None, span));
    let entry = typed_function(
        match entry_kind {
            Entry::Program => ENTRY,
            Entry::Host => crate::HOST_ENTRY,
        },
        Vec::new(),
        prim(PrimitiveType::Unit),
        entry_body,
        span,
    );
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(chunk_fn),
        Type::Unknown,
        span,
    ));
    declarations.push(TypedNode::new(
        TypedDeclaration::Function(entry),
        Type::Unknown,
        span,
    ));
    for hook in hooks {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(hook),
            Type::Unknown,
            span,
        ));
    }
    // The library itself arrives by import: its declarations for
    // typing, its HIR to link against.
    declarations.push(library_import());
    let mut source_files = vec![zyntax_typed_ast::source::SourceFile::new(
        file.to_string(),
        source.to_string(),
    )];
    for m in &loaded {
        source_files.push(zyntax_typed_ast::source::SourceFile::new(
            m.file.clone(),
            m.source.clone(),
        ));
    }
    let mut program = TypedProgram {
        checker_interface: None,
        declarations,
        language: Some(intern("lua")),
        span,
        source_files,
        type_registry: registry,
    };
    let named = references.finish();
    let TypedDeclaration::Import(import) = &mut program
        .declarations
        .last_mut()
        .expect("the library import")
        .node
    else {
        unreachable!("the last declaration is the library import");
    };
    import.items = named
        .into_iter()
        .map(|name| zyntax_typed_ast::typed_ast::TypedImportItem::Named { name, alias: None })
        .collect();
    Ok(program)
}
