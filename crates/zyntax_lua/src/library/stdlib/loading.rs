//! Loading declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        k,
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
    d.push(define(
        "zl_string_metatable",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_string_meta")),
                vec![
                    tb.decl(call("zl_table_new", vec![], table.clone())),
                    expr(call(
                        "zl_rawset_str",
                        vec![
                            tb.e(),
                            text("__index"),
                            call(&lib_table_fn("string"), vec![], any()),
                        ],
                        unit(),
                    )),
                    assign_global("zl_string_meta", box_table(tb.e())),
                ],
            ),
            ret(cached("zl_string_meta")),
        ],
    ));
    let is_func = |x: Expr| {
        and(
            ne(x.clone(), nil()),
            eq(tag_of(x), int(zyntax_builtins::FUNC_TAG)),
        )
    };
    d.push(define(
        "zl_string_member",
        &[x, y],
        any(),
        vec![
            handler.decl(call(
                "zl_index",
                vec![
                    call("zl_string_metatable", vec![], any()),
                    box_str(text("__index")),
                ],
                any(),
            )),
            when(is_nil(handler.e()), vec![ret(nil())]),
            when(
                is_func(handler.e()),
                vec![ret(call(
                    "zl_first",
                    vec![call("zl_call_2", vec![handler.e(), x.e(), y.e()], any())],
                    any(),
                ))],
            ),
            ret(call("zl_index", vec![handler.e(), y.e()], any())),
        ],
    ));

    // `load(chunk, name, mode, env)`: the chunk, a string or a
    // function giving pieces of it, compiled while the program runs;
    // the function value, or nil and the message.
    d.push(extern_fn(
        "zl_load_host",
        &[("source", string()), ("name", string()), ("env", any())],
        any(),
        Some("$Lua$load"),
    ));
    d.push(extern_fn(
        "zl_load_wraps",
        &[],
        boolean(),
        Some("$Lua$load_wraps"),
    ));
    d.push(extern_fn(
        "zl_dump_raw",
        &[("key", i64()), ("strip", boolean())],
        string(),
        Some("$Lua$dump"),
    ));
    let source = kept("source", string());
    // `require(name)`: what is loaded, a preloaded module (the
    // program's own files, compiled with it), or a file along
    // `package.path`.
    let name = kept("name", string());
    let env = kept("env", any());
    // A chunk compiled: its function value, or nil with the message
    // for `zl_load_error`. A binary chunk holding a function compiles
    // to a chunk returning it, which is called for it.
    d.push(define(
        "zl_load_raw",
        &[&source, &name, &env],
        any(),
        vec![
            y.decl(call(
                "zl_load_host",
                vec![source.e(), name.e(), env.e()],
                any(),
            )),
            when(
                and(not(is_nil(y.e())), call("zl_load_wraps", vec![], boolean())),
                vec![y.set(call(
                    "zl_first",
                    vec![call("zl_call_0", vec![y.e()], any())],
                    any(),
                ))],
            ),
            ret(y.e()),
        ],
    ));
    // `string.dump(f, strip)`: the binary chunk `load` reads back.
    let strip = kept("strip", any());
    d.push(define(
        "zl_string_dump",
        &[x, &strip],
        string(),
        vec![
            when(
                not(is_func(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'dump' (function expected, got "),
                    type_name(x.e()),
                    text(")"),
                ]))],
            ),
            s.decl(call(
                "zl_dump_raw",
                vec![
                    call("zl_func_id", vec![x.e()], i64()),
                    call("zl_truthy", vec![strip.e()], boolean()),
                ],
                string(),
            )),
            when(
                eq(s.e(), null(string())),
                vec![lua_error(text("unable to dump given function"))],
            ),
            ret(s.e()),
        ],
    ));
    d.push(extern_fn(
        "zl_searchpath",
        &[
            ("name", string()),
            ("path", string()),
            ("sep", string()),
            ("rep", string()),
        ],
        string(),
        Some("$Lua$searchpath"),
    ));
    d.push(extern_fn(
        "zl_load_error",
        &[],
        string(),
        Some("$Lua$load_error"),
    ));
    d.push(extern_fn(
        "zl_cpath_default",
        &[],
        string(),
        Some("$Lua$cpath"),
    ));
    let chunk = kept("chunk", any());
    let chunk_name = kept("chunk_name", any());
    let mode = kept("mode", any());
    // The arguments after `mode`: an environment when there is one,
    // nil included, which is not the same as none.
    let given = kept("given", anys.clone());
    let chunk_env = local("chunk_env", any());
    let piece = local("piece", any());
    let cname = local("cname", string());
    let env_given = |given: &Local, chunk_env: &Local| {
        when(
            gt(len(given.e()), int(0)),
            vec![chunk_env.set(call("zl_value_at", vec![given.e(), int(1)], any()))],
        )
    };
    let read_piece = || {
        block_of(vec![
            piece.set(call(
                "zl_first",
                vec![call("zl_call_0", vec![chunk.e()], any())],
                any(),
            )),
            when(
                not(is_nil(pending())),
                vec![
                    expr(call("zl_buf_close", vec![], string())),
                    ret(call(
                        "zb_box_tuple",
                        vec![list(
                            vec![nil(), call("zl_take_pending", vec![], any())],
                            anys.clone(),
                        )],
                        any(),
                    )),
                ],
            ),
            when(
                and(
                    not(is_nil(piece.e())),
                    or(
                        eq(category(piece.e()), int(INT)),
                        eq(category(piece.e()), int(FLOAT)),
                    ),
                ),
                vec![piece.set(box_str(call(
                    "zl_arg_str",
                    vec![piece.e(), text("")],
                    string(),
                )))],
            ),
        ])
    };
    d.push(define(
        "zl_load",
        &[&chunk, &chunk_name, &mode, &given],
        any(),
        vec![
            s.decl(text("")),
            // A chunk is named for its text, or `=(load)` when a reader
            // gives it, unless the call names it.
            cname.decl(text("=(load)")),
            if_(
                and(
                    not(is_nil(chunk.e())),
                    or(
                        eq(category(chunk.e()), int(STR)),
                        or(
                            eq(category(chunk.e()), int(INT)),
                            or(
                                eq(category(chunk.e()), int(UINT)),
                                eq(category(chunk.e()), int(FLOAT)),
                            ),
                        ),
                    ),
                ),
                vec![
                    s.set(call("zl_arg_str", vec![chunk.e(), text("")], string())),
                    cname.set(s.e()),
                ],
                vec![
                    when(
                        not(is_func(chunk.e())),
                        vec![lua_error(concat(vec![
                            text("bad argument #1 to 'load' (function expected, got "),
                            arg_type_name(chunk.e()),
                            text(")"),
                        ]))],
                    ),
                    // The reader gives pieces until nil or an empty string;
                    // a number is its string. An error it raises is what
                    // `load` returns.
                    expr(call("zl_buf_open", vec![], unit())),
                    piece.decl(nil()),
                    read_piece(),
                    while_(
                        and(
                            not(is_nil(piece.e())),
                            and(
                                eq(category(piece.e()), int(STR)),
                                gt(call("zb_str_len", vec![get_str(piece.e())], i64()), int(0)),
                            ),
                        ),
                        vec![
                            expr(call("zl_buf_push", vec![get_str(piece.e())], unit())),
                            read_piece(),
                        ],
                    ),
                    s.set(call("zl_buf_close", vec![], string())),
                    when(
                        and(not(is_nil(piece.e())), ne(category(piece.e()), int(STR))),
                        vec![ret(call(
                            "zb_box_tuple",
                            vec![list(
                                vec![nil(), box_str(text("reader function must return a string"))],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                ],
            ),
            when(
                eq(category(chunk_name.e()), int(STR)),
                vec![cname.set(get_str(chunk_name.e()))],
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
                vec![s.e(), cname.e(), chunk_env.e()],
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
    // What `mode` says about a chunk: a chunk starting with an escape
    // byte is binary, which needs `b`; a text chunk needs `t`. The
    // message when the chunk is refused, or nil.
    d.push(define(
        "zl_load_mode_error",
        &[s, &mode],
        any(),
        vec![
            // The first byte: a binary chunk is not UTF-8 past its header.
            k.decl(call("zl_byte_at", vec![s.e(), int(1)], i64())),
            cname.decl(text("bt")),
            when(
                and(not(is_nil(mode.e())), eq(category(mode.e()), int(STR))),
                vec![cname.set(get_str(mode.e()))],
            ),
            when(
                and(
                    eq(k.e(), int(27)),
                    not(call(
                        "zb_str_contains",
                        vec![cname.e(), text("b")],
                        boolean(),
                    )),
                ),
                vec![ret(box_str(concat(vec![
                    text("attempt to load a binary chunk (mode is '"),
                    cname.e(),
                    text("')"),
                ])))],
            ),
            when(
                and(
                    ne(k.e(), int(27)),
                    not(call(
                        "zb_str_contains",
                        vec![cname.e(), text("t")],
                        boolean(),
                    )),
                ),
                vec![ret(box_str(concat(vec![
                    text("attempt to load a text chunk (mode is '"),
                    cname.e(),
                    text("')"),
                ])))],
            ),
            ret(nil()),
        ],
    ));
    // `loadfile(name, mode, env)`: the file as a chunk named `@name`.
    d.push(extern_fn(
        "zl_read_file",
        &[("path", string())],
        string(),
        Some("$Lua$read_file"),
    ));
}
