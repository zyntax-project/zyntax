//! Lines declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        a,
        anys,
        args,
        env,
        h,
        i,
        name,
        packed,
        tb,
        vals,
        x,
        y,
        ..
    } = context;

    // ─── files as values ────────────────────────────────────────
    let handle_of = |x: Expr| call("zb_unbox_instance_raw", vec![x], i64());
    let is_open = |x: Expr| call("zl_io_is_open", vec![handle_of(x)], boolean());

    // ─── lines ──────────────────────────────────────────────────
    // `f:lines(...)` and `io.lines(name, ...)`: a function reading the
    // formats each call; at the end it closes the file when it opened
    // it, and a read that fails is an error.
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    d.push(define(
        "zl_io_lines_code",
        &[env, packed],
        any(),
        vec![
            x.decl(at(env.e(), int(2))),
            when(
                not(is_open(x.e())),
                vec![lua_error(text("file is already closed"))],
            ),
            y.decl(call(
                "zl_file_read",
                vec![
                    x.e(),
                    call("zl_values", vec![at(env.e(), int(3))], anys.clone()),
                ],
                any(),
            )),
            when(not(is_nil(pending())), vec![ret(nil())]),
            vals.decl(call("zl_values", vec![y.e()], anys.clone())),
            when(
                and(gt(len(vals.e()), int(0)), not(is_nil(at(vals.e(), int(0))))),
                vec![ret(y.e())],
            ),
            // nil, a message and a number: the read failed.
            when(
                and(
                    eq(len(vals.e()), int(3)),
                    and(
                        not(is_nil(at(vals.e(), int(1)))),
                        eq(category(at(vals.e(), int(1))), int(STR)),
                    ),
                ),
                vec![lua_error(get_str(at(vals.e(), int(1))))],
            ),
            when(
                get_bool(at(env.e(), int(4))),
                vec![expr(call("zl_file_release", vec![x.e()], any()))],
            ),
            ret(call("zl_none", vec![], any())),
        ],
    ));
    let closing = local("closing", boolean());
    d.push(define(
        "zl_file_lines",
        &[x, args, &closing],
        any(),
        vec![
            expr(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "lines")],
                i64(),
            )),
            // The reference keeps the formats on its stack: 250 at most.
            when(
                gt(len(args.e()), int(250)),
                vec![lua_error(text(
                    "bad argument #252 to 'lines' (too many arguments)",
                ))],
            ),
            // Each format is checked now, as the reference checks them.
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    a.decl(at(args.e(), i.e())),
                    when(
                        or(
                            is_nil(a.e()),
                            not(or(
                                eq(category(a.e()), int(STR)),
                                or(
                                    eq(category(a.e()), int(INT)),
                                    or(
                                        eq(category(a.e()), int(UINT)),
                                        eq(category(a.e()), int(FLOAT)),
                                    ),
                                ),
                            )),
                        ),
                        vec![lua_error(concat(vec![
                            bad_arg_at(add(i.e(), int(1)), "lines"),
                            text(" (invalid format)"),
                        ]))],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            y.decl(call(
                "zb_func_new",
                vec![
                    code_of("zl_io_lines_code"),
                    int(zyntax_builtins::functions::VARIADIC_ARITY),
                    list(
                        vec![
                            x.e(),
                            call("zb_box_tuple", vec![args.e()], any()),
                            box_bool(closing.e()),
                        ],
                        anys.clone(),
                    ),
                ],
                any(),
            )),
            // A file `io.lines` opened is the loop's closing value; any
            // other gives the iterator alone.
            when(not(closing.e()), vec![ret(y.e())]),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![y.e(), nil(), nil(), x.e()], anys.clone())],
                any(),
            )),
        ],
    ));
    // `io.lines([name, ...])`.
    d.push(define(
        "zl_io_lines",
        &[x, args],
        any(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(call(
                    "zl_file_lines",
                    vec![
                        call("zl_io_default_in", vec![], any()),
                        args.e(),
                        bool(false),
                    ],
                    any(),
                ))],
            ),
            name.decl(call(
                "zl_arg_str",
                vec![x.e(), bad_arg(1, "lines")],
                string(),
            )),
            h.decl(call("zl_io_open", vec![name.e(), text("r")], i64())),
            when(
                eq(h.e(), int(0)),
                vec![lua_error(concat(vec![
                    text("cannot open file '"),
                    name.e(),
                    text("' ("),
                    call("zl_io_reason", vec![], string()),
                    text(")"),
                ]))],
            ),
            ret(call(
                "zl_file_lines",
                vec![
                    call("zl_file_new", vec![h.e()], any()),
                    args.e(),
                    bool(true),
                ],
                any(),
            )),
        ],
    ));
    // `f:lines(...)` as the method: the file stays open at the end.
    d.push(define(
        "zl_file_lines_of",
        &[x, args],
        any(),
        vec![ret(call(
            "zl_file_lines",
            vec![x.e(), args.e(), bool(false)],
            any(),
        ))],
    ));
    d.push(define(
        "zl_file_tostring",
        &[x],
        string(),
        vec![
            when(
                not(is_file(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to '__tostring' (FILE* expected, got "),
                    arg_type_name(x.e()),
                    text(")"),
                ]))],
            ),
            ret(call("zl_file_str", vec![x.e()], string())),
        ],
    ));
    // `io.read(...)`, `io.write(...)`, `io.close([f])`, `io.flush()`.
    d.push(define(
        "zl_io_read",
        &[args],
        any(),
        vec![ret(call(
            "zl_file_read",
            vec![call("zl_io_default_in", vec![], any()), args.e()],
            any(),
        ))],
    ));
    d.push(define(
        "zl_io_write_of",
        &[args],
        any(),
        vec![ret(call(
            "zl_file_write",
            vec![call("zl_io_default_out", vec![], any()), args.e()],
            any(),
        ))],
    ));
    d.push(define(
        "zl_io_close_of",
        &[x],
        any(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(call(
                    "zl_file_close",
                    vec![call("zl_io_output_file", vec![], any())],
                    any(),
                ))],
            ),
            ret(call("zl_file_close", vec![x.e()], any())),
        ],
    ));
    d.push(define(
        "zl_io_flush_of",
        &[],
        any(),
        vec![ret(call(
            "zl_file_flush",
            vec![call("zl_io_default_out", vec![], any())],
            any(),
        ))],
    ));

    // ─── the metatable ──────────────────────────────────────────
    // `__index` is the methods table, the metamethods and `__name`
    // sit beside it, as the reference lays them out.
    d.push(global_var("zl_file_meta", any()));
    d.push(define(
        "zl_file_metatable",
        &[],
        any(),
        vec![
            when(
                is_nil(read_global("zl_file_meta", any())),
                vec![
                    tb.decl(unbox_table(
                        call(&stdlib::lib_table_fn("filemeta"), vec![], any()),
                        t,
                    )),
                    expr(call(
                        "zl_rawset_str",
                        vec![
                            tb.e(),
                            text("__index"),
                            call(&stdlib::lib_table_fn("file"), vec![], any()),
                        ],
                        unit(),
                    )),
                    expr(call(
                        "zl_rawset_str",
                        vec![tb.e(), text("__name"), box_str(text("FILE*"))],
                        unit(),
                    )),
                    set_global("zl_file_meta", box_table(tb.e())),
                ],
            ),
            ret(read_global("zl_file_meta", any())),
        ],
    ));
    // `f.name`: the method, through the metatable's `__index`.
    d.push(define(
        "zl_file_member",
        &[x, y],
        any(),
        vec![
            a.decl(call(
                "zl_index",
                vec![
                    call("zl_file_metatable", vec![], any()),
                    box_str(text("__index")),
                ],
                any(),
            )),
            when(is_nil(a.e()), vec![ret(nil())]),
            when(
                is_func(a.e()),
                vec![ret(call(
                    "zl_first",
                    vec![call("zl_call_2", vec![a.e(), x.e(), y.e()], any())],
                    any(),
                ))],
            ),
            ret(call("zl_index", vec![a.e(), y.e()], any())),
        ],
    ));
}
