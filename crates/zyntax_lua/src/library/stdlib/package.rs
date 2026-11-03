//! Package declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        acc,
        anys,
        s,
        table,
        tb,
        x,
        y,
        ..
    } = context;
    // `pcall(f, ...)`: true and the results, or false and the error.
    // The call is from no line: what it calls directly reports no
    // position, as under the reference where the caller is C.
    let handler = kept("handler", any());
    let cached = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            any(),
        )
    };
    let assign_global = |name: &str, value: Expr| {
        expr(node(
            zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                zyntax_typed_ast::typed_ast::TypedBinary {
                    op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                    left: Box::new(cached(name)),
                    right: Box::new(value),
                },
            ),
            any(),
        ))
    };
    let mode = kept("mode", any());
    // The arguments after `mode`: an environment when there is one,
    // nil included, which is not the same as none.
    let given = kept("given", anys.clone());
    let chunk_env = local("chunk_env", any());
    let env_given = |given: &Local, chunk_env: &Local| {
        when(
            gt(len(given.e()), int(0)),
            vec![chunk_env.set(call("zl_value_at", vec![given.e(), int(1)], any()))],
        )
    };
    let path = kept("path", string());
    d.push(define(
        "zl_loadfile",
        &[&path, &mode, &given],
        any(),
        vec![
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            y.decl(call("zl_load_mode_error", vec![s.e(), mode.e()], any())),
            when(
                not(is_nil(y.e())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![nil(), y.e()], anys.clone())],
                    any(),
                ))],
            ),
            chunk_env.decl(call("zl_globals_value", vec![], any())),
            env_given(&given, &chunk_env),
            y.set(call(
                "zl_load_raw",
                vec![s.e(), add(text("@"), path.e()), chunk_env.e()],
                any(),
            )),
            when(
                is_nil(y.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(y.e()),
        ],
    ));
    // `dofile(name)`: the file run, its values returned; an error in
    // loading is raised.
    d.push(define(
        "zl_dofile",
        &[&path],
        any(),
        vec![
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            y.decl(call(
                "zl_load_raw",
                vec![
                    s.e(),
                    add(text("@"), path.e()),
                    call("zl_globals_value", vec![], any()),
                ],
                any(),
            )),
            when(
                is_nil(y.e()),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            ret(call("zl_call_0", vec![y.e()], any())),
        ],
    ));
    // `require(name)`: what is loaded, a preloaded module (the
    // program's own files, compiled with it), or a file along
    // `package.path`.
    let name = kept("name", string());
    // A loader found, called with the name and `data`; the module it
    // gives is entered and returned with the data.
    let loader_run = |data: Expr| {
        vec![
            path.decl(data),
            y.set(call(
                "zl_first",
                vec![call(
                    "zl_call_2",
                    vec![handler.e(), box_str(name.e()), box_str(path.e())],
                    any(),
                )],
                any(),
            )),
            when(not(is_nil(pending())), vec![ret(nil())]),
            when(is_nil(y.e()), vec![y.set(box_bool(bool(true)))]),
            expr(call(
                "zl_rawset_str",
                vec![
                    unbox_table(call("zl_package_loaded", vec![], any()), t),
                    name.e(),
                    y.e(),
                ],
                unit(),
            )),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![y.e(), box_str(path.e())], anys.clone())],
                any(),
            )),
        ]
    };
    let cpath = kept("cpath", string());
    let file = kept("file", string());
    let cmsg = kept("cmsg", string());
    let status = local("status", i64());
    let dot = local("dot", i64());
    // A C library found whose open function would not load: the error
    // `require` raises for it.
    let load_failed = |file: &Local| {
        lua_error(concat(vec![
            text("error loading module '"),
            name.e(),
            text("' from file '"),
            file.e(),
            text("':\n\t"),
            call("zl_c_error", vec![], string()),
        ]))
    };
    d.push(define(
        "zl_require",
        &[&name],
        any(),
        vec![
            y.decl(call(
                "zl_index",
                vec![call("zl_package_loaded", vec![], any()), box_str(name.e())],
                any(),
            )),
            // Loaded already: the value alone. Loaded now: the value
            // and what the loader was given, the file it came from.
            when(
                call("zl_truthy", vec![y.e()], boolean()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![y.e()], anys.clone())],
                    any(),
                ))],
            ),
            when(
                not(is_table(call(
                    "zl_index",
                    vec![
                        call(&lib_table_fn("package"), vec![], any()),
                        box_str(text("searchers")),
                    ],
                    any(),
                ))),
                vec![lua_error(text("'package.searchers' must be a table"))],
            ),
            // One of the program's own files, entered before it ran: the
            // loader sees the name and the file, `./name.lua` under the
            // program's directory. Then a loader in `package.preload`,
            // which sees `:preload:`. What either returns is the
            // module, or true when it returns nothing.
            handler.decl(call(
                "zl_index",
                vec![call("zl_static_modules", vec![], any()), box_str(name.e())],
                any(),
            )),
            when(
                not(is_nil(handler.e())),
                loader_run(concat(vec![
                    text("./"),
                    call("zl_module_path", vec![name.e()], string()),
                    text(".lua"),
                ])),
            ),
            handler.set(call(
                "zl_index",
                vec![call("zl_package_preload", vec![], any()), box_str(name.e())],
                any(),
            )),
            when(not(is_nil(handler.e())), loader_run(text(":preload:"))),
            // A file along `package.path`, compiled now and run as the
            // loader with the name and its file.
            handler.decl(call(
                "zl_index",
                vec![
                    call(&lib_table_fn("package"), vec![], any()),
                    box_str(text("path")),
                ],
                any(),
            )),
            when(
                or(
                    is_nil(handler.e()),
                    and(
                        ne(category(handler.e()), int(STR)),
                        and(
                            ne(category(handler.e()), int(INT)),
                            and(
                                ne(category(handler.e()), int(UINT)),
                                ne(category(handler.e()), int(FLOAT)),
                            ),
                        ),
                    ),
                ),
                vec![lua_error(text("'package.path' must be a string"))],
            ),
            path.decl(call(
                "zl_searchpath",
                vec![
                    name.e(),
                    call("zl_arg_str", vec![handler.e(), text("")], string()),
                    text("."),
                    text("/"),
                ],
                string(),
            )),
            // Not there: a C library along `package.cpath` whose open
            // function is the module's, then, for a dotted name, one
            // named after the name's root that holds the module's open
            // function among others.
            when(
                eq(path.e(), null(string())),
                vec![
                    acc.decl(call("zl_load_error", vec![], string())),
                    handler.set(call(
                        "zl_index",
                        vec![
                            call(&lib_table_fn("package"), vec![], any()),
                            box_str(text("cpath")),
                        ],
                        any(),
                    )),
                    when(
                        or(
                            is_nil(handler.e()),
                            and(
                                ne(category(handler.e()), int(STR)),
                                and(
                                    ne(category(handler.e()), int(INT)),
                                    and(
                                        ne(category(handler.e()), int(UINT)),
                                        ne(category(handler.e()), int(FLOAT)),
                                    ),
                                ),
                            ),
                        ),
                        vec![lua_error(text("'package.cpath' must be a string"))],
                    ),
                    cpath.decl(call("zl_arg_str", vec![handler.e(), text("")], string())),
                    file.decl(call(
                        "zl_searchpath",
                        vec![name.e(), cpath.e(), text("."), text("/")],
                        string(),
                    )),
                    when(ne(file.e(), null(string())), {
                        let mut st = vec![
                            status.decl(call("zl_c_loadfunc", vec![file.e(), name.e()], i64())),
                            when(ne(status.e(), int(0)), vec![load_failed(&file)]),
                            handler.set(call("zl_c_loaded", vec![], any())),
                        ];
                        st.extend(loader_run(file.e()));
                        st
                    }),
                    cmsg.decl(call("zl_load_error", vec![], string())),
                    dot.decl(call(
                        "zl_find_plain",
                        vec![name.e(), text("."), int(1)],
                        i64(),
                    )),
                    when(
                        gt(dot.e(), int(0)),
                        vec![
                            file.set(call(
                                "zl_searchpath",
                                vec![
                                    call(
                                        "zl_string_sub",
                                        vec![name.e(), int(1), sub(dot.e(), int(1))],
                                        string(),
                                    ),
                                    cpath.e(),
                                    text("."),
                                    text("/"),
                                ],
                                string(),
                            )),
                            if_(
                                ne(file.e(), null(string())),
                                vec![
                                    status.decl(call(
                                        "zl_c_loadfunc",
                                        vec![file.e(), name.e()],
                                        i64(),
                                    )),
                                    when(eq(status.e(), int(0)), {
                                        let mut st =
                                            vec![handler.set(call("zl_c_loaded", vec![], any()))];
                                        st.extend(loader_run(file.e()));
                                        st
                                    }),
                                    when(eq(status.e(), int(1)), vec![load_failed(&file)]),
                                    cmsg.set(concat(vec![
                                        cmsg.e(),
                                        text("\n\tno module '"),
                                        name.e(),
                                        text("' in file '"),
                                        file.e(),
                                        text("'"),
                                    ])),
                                ],
                                vec![cmsg.set(concat(vec![
                                    cmsg.e(),
                                    text("\n\t"),
                                    call("zl_load_error", vec![], string()),
                                ]))],
                            ),
                        ],
                    ),
                    // Nor there: a module of the embedder's, a foreign object.
                    y.set(call("zb_foreign_import", vec![name.e()], any())),
                    when(not(is_nil(pending())), vec![ret(nil())]),
                    when(
                        not(is_nil(y.e())),
                        vec![
                            expr(call(
                                "zl_rawset_str",
                                vec![
                                    unbox_table(call("zl_package_loaded", vec![], any()), t),
                                    name.e(),
                                    y.e(),
                                ],
                                unit(),
                            )),
                            ret(call(
                                "zb_box_tuple",
                                vec![list(vec![y.e(), box_str(text(":foreign:"))], anys.clone())],
                                any(),
                            )),
                        ],
                    ),
                    lua_error(concat(vec![
                        text("module '"),
                        name.e(),
                        text("' not found:\n\tno field package.preload['"),
                        name.e(),
                        text("']\n\t"),
                        acc.e(),
                        text("\n\t"),
                        cmsg.e(),
                    ])),
                ],
            ),
            s.decl(call("zl_read_file", vec![path.e()], string())),
            when(
                eq(s.e(), null(string())),
                vec![lua_error(call("zl_load_error", vec![], string()))],
            ),
            handler.set(call(
                "zl_load_raw",
                vec![
                    s.e(),
                    add(text("@"), path.e()),
                    call("zl_globals_value", vec![], any()),
                ],
                any(),
            )),
            when(
                is_nil(handler.e()),
                vec![lua_error(concat(vec![
                    text("error loading module '"),
                    name.e(),
                    text("' from file '"),
                    path.e(),
                    text("':\n\t"),
                    call("zl_load_error", vec![], string()),
                ]))],
            ),
            y.set(call(
                "zl_first",
                vec![call(
                    "zl_call_2",
                    vec![handler.e(), box_str(name.e()), box_str(path.e())],
                    any(),
                )],
                any(),
            )),
            when(not(is_nil(pending())), vec![ret(nil())]),
            // The module may have set its own entry while loading.
            when(
                is_nil(y.e()),
                vec![y.set(call(
                    "zl_index",
                    vec![call("zl_package_loaded", vec![], any()), box_str(name.e())],
                    any(),
                ))],
            ),
            when(is_nil(y.e()), vec![y.set(box_bool(bool(true)))]),
            expr(call(
                "zl_rawset_str",
                vec![
                    unbox_table(call("zl_package_loaded", vec![], any()), t),
                    name.e(),
                    y.e(),
                ],
                unit(),
            )),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![y.e(), box_str(path.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    // `package.loadlib(path, name)`: the C function `name` of the
    // library at `path`, or true for `*`, which makes its symbols
    // global; else nil, the message, and whether the library would not
    // open or has no such function.
    d.push(define(
        "zl_package_loadlib",
        &[&path, &name],
        any(),
        vec![
            status.decl(call("zl_c_loadlib", vec![path.e(), name.e()], i64())),
            when(
                eq(status.e(), int(0)),
                vec![ret(call("zl_c_loaded", vec![], any()))],
            ),
            when(
                eq(status.e(), int(1)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            nil(),
                            box_str(call("zl_c_error", vec![], string())),
                            box_str(text("open")),
                        ],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        nil(),
                        box_str(call("zl_c_error", vec![], string())),
                        box_str(text("init")),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    // `package.searchpath(name, path [, sep [, rep]])`: the file found,
    // or nil and the list of files tried.
    let sep = kept("sep", string());
    let rep = kept("rep", string());
    d.push(define(
        "zl_package_searchpath",
        &[&name, s, &sep, &rep],
        any(),
        vec![
            path.decl(call(
                "zl_searchpath",
                vec![name.e(), s.e(), sep.e(), rep.e()],
                string(),
            )),
            when(
                eq(path.e(), null(string())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(call("zl_load_error", vec![], string()))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_str(path.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    // `package.preload`: loaders by module name; the program's own
    // files are entered here before it runs.
    d.push(global_var("zl_preload", any()));
    d.push(define(
        "zl_package_preload",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_preload")),
                vec![assign_global(
                    "zl_preload",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            ret(cached("zl_preload")),
        ],
    ));
    // A module name's path: its dots as directory separators.
    d.push(define(
        "zl_module_path",
        &[&name],
        string(),
        vec![ret(call("zl_buf_replace_dots", vec![name.e()], string()))],
    ));
    // The program's own files as loaders, apart from `package.preload`
    // so a program sees only what it put there.
    d.push(global_var("zl_static", any()));
    d.push(define(
        "zl_static_modules",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_static")),
                vec![assign_global(
                    "zl_static",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            ret(cached("zl_static")),
        ],
    ));
    d.push(define(
        "zl_preload_module",
        &[&name, x],
        unit(),
        vec![
            expr(call(
                "zl_rawset_str",
                vec![
                    unbox_table(call("zl_static_modules", vec![], any()), t),
                    name.e(),
                    x.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // `package.loaded`: every library under its name, and `_G`.
    d.push(global_var("zl_loaded", any()));
    {
        let mut st = vec![when(
            not(is_nil(cached("zl_loaded"))),
            vec![ret(cached("zl_loaded"))],
        )];
        st.push(tb.decl(call("zl_table_new", vec![], table.clone())));
        for lib in LIBS {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(lib), call(&lib_table_fn(lib), vec![], any())],
                unit(),
            )));
        }
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_G"), call("zl_globals_value", vec![], any())],
            unit(),
        )));
        st.push(assign_global("zl_loaded", box_table(tb.e())));
        st.push(ret(cached("zl_loaded")));
        d.push(define("zl_package_loaded", &[], any(), st));
    }
    // `_G` as a value for a program that never reaches it as one.
    d.push(define(
        "zl_globals_value",
        &[],
        any(),
        vec![ret(box_table(call(
            "zl_globals_table",
            vec![],
            table.clone(),
        )))],
    ));
}
