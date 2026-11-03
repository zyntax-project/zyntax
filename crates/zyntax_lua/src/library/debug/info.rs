//! Info declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        first,
        i,
        j,
        k,
        key,
        kind,
        level,
        lines,
        n,
        name,
        table,
        tb,
        thread,
        what,
        x,
        y,
        z,
        ..
    } = context;
    let line_now = || read_global(LINE, i64());
    let thread_handle = |co: Expr| {
        call(
            "zb_box_get_i64",
            vec![at(
                call("zb_unbox_list_raw_any", vec![co], anys.clone()),
                int(0),
            )],
            i64(),
        )
    };

    // ─── the library ────────────────────────────────────────────
    // The thread a debug function is asked about, when its first
    // argument is one: its handle and 1 (the arguments after it
    // start one later); else -1 and 0.
    let thread_arg = |args: &Local, thread: &Local, first: &Local| {
        vec![
            thread.decl(int(-1)),
            first.decl(int(0)),
            x.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            when(
                is_thread(x.e()),
                vec![thread.set(thread_handle(x.e())), first.set(int(1))],
            ),
        ]
    };
    let arg = |i: Expr| call("zl_value_at", vec![args.e(), add(first.e(), i)], any());

    // What an argument error calls the function `what`: `debug.what`
    // when it was called as a value.
    let qual = local("qual", boolean());
    let fname = |what: &str| if_expr(qual.e(), text(&format!("debug.{what}")), text(what));
    let bad = |n: usize, what: &str, rest: &str| {
        concat(vec![
            text(&format!("bad argument #{n} to '")),
            fname(what),
            text(&format!("'{rest}")),
        ])
    };
    let qualify = || {
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
        ]
    };

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(args, thread, first));
    st.extend([
        y.decl(arg(int(1))),
        when(
            and(
                not(is_nil(y.e())),
                and(
                    ne(category(y.e()), int(STR)),
                    and(
                        ne(category(y.e()), int(INT)),
                        and(
                            ne(category(y.e()), int(UINT)),
                            ne(category(y.e()), int(FLOAT)),
                        ),
                    ),
                ),
            ),
            vec![ret(y.e())],
        ),
        level.decl(if_expr(eq(thread.e(), int(-1)), int(1), int(0))),
        z.decl(arg(int(2))),
        when(
            not(is_nil(z.e())),
            vec![level.set(call(
                "zl_arg_int",
                vec![z.e(), bad(2, "traceback", "")],
                i64(),
            ))],
        ),
        name.decl(text("")),
        when(
            not(is_nil(y.e())),
            vec![name.set(call("zl_tostring", vec![y.e()], string()))],
        ),
        ret(box_str(call(
            "zl_dbg_traceback_raw",
            vec![
                name.e(),
                not(is_nil(y.e())),
                level.e(),
                thread.e(),
                line_now(),
            ],
            string(),
        ))),
    ]);
    d.push(define("zl_debug_traceback", &[args], any(), st));

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(args, thread, first));
    st.extend([
        y.decl(arg(int(1))),
        z.decl(arg(int(2))),
        what.decl(text("flnSrtu")),
        when(
            not(is_nil(z.e())),
            vec![
                when(
                    and(
                        ne(category(z.e()), int(STR)),
                        is_nil(call("zl_arith_operand", vec![z.e()], any())),
                    ),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                        text(" to '"),
                        fname("getinfo"),
                        text("' (string expected, got "),
                        arg_type_name(z.e()),
                        text(")"),
                    ]))],
                ),
                what.set(call("zl_tostring", vec![z.e()], string())),
            ],
        ),
        kind.decl(call("zl_dbg_check_options", vec![what.e()], i64())),
        when(
            ne(kind.e(), int(0)),
            vec![lua_error(concat(vec![
                text("bad argument #"),
                call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                text(" to '"),
                fname("getinfo"),
                text("' (invalid option"),
                if_expr(eq(kind.e(), int(2)), text(" '>')"), text(")")),
            ]))],
        ),
        if_(
            is_func(y.e()),
            vec![expr(call(
                "zl_dbg_info_func",
                vec![call("zl_func_id", vec![y.e()], i64()), what.e()],
                unit(),
            ))],
            vec![
                when(
                    lt(len(args.e()), add(first.e(), int(1))),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(1))], string()),
                        text(" to '"),
                        fname("getinfo"),
                        text("' (number expected, got no value)"),
                    ]))],
                ),
                level.decl(call(
                    "zl_arg_int",
                    vec![
                        y.e(),
                        concat(vec![
                            text("bad argument #"),
                            call("zb_str_of_int", vec![add(first.e(), int(1))], string()),
                            text(" to '"),
                            fname("getinfo"),
                            text("'"),
                        ]),
                    ],
                    i64(),
                )),
                when(
                    eq(
                        call(
                            "zl_dbg_info_level",
                            vec![level.e(), thread.e(), line_now(), what.e()],
                            i64(),
                        ),
                        int(0),
                    ),
                    vec![ret(nil())],
                ),
            ],
        ),
        tb.decl(call("zl_table_new", vec![], table.clone())),
        n.decl(call("zl_dbg_info_count", vec![], i64())),
        i.decl(int(0)),
        while_(
            lt(i.e(), n.e()),
            vec![
                name.decl(call("zl_dbg_info_name", vec![i.e()], string())),
                kind.set(call("zl_dbg_info_kind", vec![i.e()], i64())),
                x.set(nil()),
                when(
                    eq(kind.e(), int(1)),
                    vec![x.set(box_i64(call("zl_dbg_info_int", vec![i.e()], i64())))],
                ),
                when(
                    eq(kind.e(), int(2)),
                    vec![x.set(box_str(call("zl_dbg_info_str", vec![i.e()], string())))],
                ),
                when(
                    eq(kind.e(), int(3)),
                    vec![x.set(box_bool(ne(
                        call("zl_dbg_info_int", vec![i.e()], i64()),
                        int(0),
                    )))],
                ),
                when(
                    eq(kind.e(), int(4)),
                    vec![if_(
                        is_func(y.e()),
                        vec![x.set(y.e())],
                        vec![
                            key.decl(call("zl_dbg_info_key", vec![], i64())),
                            when(
                                and(ne(key.e(), int(-1)), not(is_nil(read_global(FUNCS, any())))),
                                vec![x.set(call(
                                    "zl_rawgeti",
                                    vec![unbox_table(read_global(FUNCS, any()), t), key.e()],
                                    any(),
                                ))],
                            ),
                        ],
                    )],
                ),
                when(
                    eq(kind.e(), int(5)),
                    vec![
                        lines.decl(call("zl_table_new", vec![], table.clone())),
                        k.decl(call("zl_dbg_info_int", vec![i.e()], i64())),
                        j.decl(int(0)),
                        while_(
                            lt(j.e(), k.e()),
                            vec![
                                expr(call(
                                    "zl_rawseti",
                                    vec![
                                        lines.e(),
                                        call("zl_dbg_info_line", vec![i.e(), j.e()], i64()),
                                        box_bool(bool(true)),
                                    ],
                                    unit(),
                                )),
                                j.add_assign(int(1)),
                            ],
                        ),
                        x.set(box_table(lines.e())),
                    ],
                ),
                when(
                    not(is_nil(x.e())),
                    vec![expr(call(
                        "zl_rawset_str",
                        vec![tb.e(), name.e(), x.e()],
                        unit(),
                    ))],
                ),
                i.add_assign(int(1)),
            ],
        ),
        ret(box_table(tb.e())),
    ]);
    d.push(define("zl_debug_getinfo", &[args], any(), st));
}
