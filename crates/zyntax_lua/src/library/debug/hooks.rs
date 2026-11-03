//! Hooks declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        f,
        first,
        k,
        name,
        thread,
        x,
        y,
        z,
        ..
    } = context;
    let line_now = || read_global(LINE, i64());
    let mask = || read_global(HOOK_MASK, i64());
    let has_event = |bit: i64| ne(bitand(mask(), int(bit)), int(0));
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
    let function_arg = |x: &Local, what: &str, number: usize| {
        when(
            not(is_func(x.e())),
            vec![lua_error(concat(vec![
                bad(number, what, " (function expected, got "),
                arg_type_name(x.e()),
                text(")"),
            ]))],
        )
    };
    let int_arg = |x: Expr, what: &str, number: usize| {
        call("zl_arg_int", vec![x, bad(number, what, "")], i64())
    };
    let m = local("m", i64());

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(args, thread, first));
    let other = local("other", boolean());
    // The thread argument `x`'s record, and whether it is dead: its
    // handle may then be another thread's.
    let thread_record = || call("zb_unbox_list_raw_any", vec![x.e()], anys.clone());
    let thread_dead = || {
        eq(
            call(
                "zb_box_get_i64",
                vec![at(thread_record(), int(coroutines::STATUS))],
                i64(),
            ),
            int(coroutines::DEAD),
        )
    };
    // A hook for another thread is set as the running one's, then
    // kept as that thread's (a dead one's in its record) while the
    // running one's comes back.
    let other_tail = || {
        when(
            other.e(),
            vec![
                if_(
                    thread_dead(),
                    vec![if_(
                        gt(len(thread_record()), int(coroutines::DEAD_HOOK)),
                        vec![set_idx(
                            thread_record(),
                            int(coroutines::DEAD_HOOK),
                            call("zl_dbg_hook_pack", vec![], any()),
                        )],
                        vec![expr(call(
                            "zb_list_extend_any",
                            vec![
                                thread_record(),
                                list(vec![call("zl_dbg_hook_pack", vec![], any())], anys.clone()),
                            ],
                            unit(),
                        ))],
                    )],
                    vec![expr(call("zl_dbg_hook_save", vec![thread.e()], unit()))],
                ),
                expr(call(
                    "zl_dbg_hook_load",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
            ],
        )
    };
    let has_char = |s: Expr, c: &str| ge(call("zb_str_index_of", vec![s, text(c)], i64()), int(0));
    // The hook is told of `sethook`'s own call and return, as of any
    // function's: its call under the hook it replaces, its return
    // under the one it sets.
    let own_event = |bit: i64, event: &str| {
        when(
            has_event(bit),
            vec![
                expr(call(
                    "zl_dbg_push",
                    vec![int(-1), line_now(), int(0)],
                    unit(),
                )),
                expr(call("zl_dbg_hook", vec![text(event), nil()], unit())),
                set_global(LINE, call("zl_dbg_pop", vec![], i64())),
            ],
        )
    };
    st.extend([
        own_event(HOOK_CALL, "call"),
        other.decl(and(
            ne(thread.e(), int(-1)),
            ne(thread.e(), read_global(HOOK_THREAD, i64())),
        )),
        when(
            other.e(),
            vec![expr(call(
                "zl_dbg_hook_save",
                vec![read_global(HOOK_THREAD, i64())],
                unit(),
            ))],
        ),
        f.decl(arg(int(1))),
        y.decl(arg(int(2))),
        z.decl(arg(int(3))),
        when(
            or(is_nil(f.e()), is_nil(y.e())),
            vec![
                set_global(HOOK_MASK, int(0)),
                set_global(HOOK_FN, nil()),
                set_global(HOOK_COUNT_EVERY, int(0)),
                other_tail(),
                ret_void(),
            ],
        ),
        function_arg(f, "sethook", 1),
        name.decl(call(
            "zl_arg_str",
            vec![y.e(), bad(2, "sethook", "")],
            string(),
        )),
        k.decl(int(0)),
        when(
            not(is_nil(z.e())),
            vec![
                k.set(int_arg(z.e(), "sethook", 3)),
                when(not(is_nil(pending())), vec![ret_void()]),
            ],
        ),
        m.decl(int(0)),
        when(
            has_char(name.e(), "c"),
            vec![m.set(bitor(m.e(), int(HOOK_CALL)))],
        ),
        when(
            has_char(name.e(), "r"),
            vec![m.set(bitor(m.e(), int(HOOK_RETURN)))],
        ),
        when(
            has_char(name.e(), "l"),
            vec![m.set(bitor(m.e(), int(HOOK_LINE)))],
        ),
        when(
            gt(k.e(), int(0)),
            vec![m.set(bitor(m.e(), int(HOOK_COUNT)))],
        ),
        set_global(HOOK_FN, f.e()),
        set_global(HOOK_COUNT_EVERY, k.e()),
        set_global(HOOK_LEFT, k.e()),
        set_global(HOOK_MASK, if_expr(eq(m.e(), int(0)), int(0), m.e())),
        when(eq(m.e(), int(0)), vec![set_global(HOOK_FN, nil())]),
        other_tail(),
        own_event(HOOK_RETURN, "return"),
        ret_void(),
    ]);
    d.push(define("zl_debug_sethook", &[args], unit(), st));
    // `debug.gethook([thread])`: the hook, its mask and its count, or
    // fail when none is set.
    let mut st = thread_arg(args, thread, first);
    st.extend([
        when(
            and(
                ne(thread.e(), int(-1)),
                ne(thread.e(), read_global(HOOK_THREAD, i64())),
            ),
            vec![
                expr(call(
                    "zl_dbg_hook_save",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
                if_(
                    thread_dead(),
                    vec![expr(call(
                        "zl_dbg_hook_use",
                        vec![if_expr(
                            gt(len(thread_record()), int(coroutines::DEAD_HOOK)),
                            at(thread_record(), int(coroutines::DEAD_HOOK)),
                            nil(),
                        )],
                        unit(),
                    ))],
                    vec![expr(call("zl_dbg_hook_load", vec![thread.e()], unit()))],
                ),
                y.decl(call("zl_dbg_gethook_now", vec![], any())),
                expr(call(
                    "zl_dbg_hook_load",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
                ret(y.e()),
            ],
        ),
        ret(call("zl_dbg_gethook_now", vec![], any())),
    ]);
    d.push(define("zl_debug_gethook", &[args], any(), st));
    // The running thread's hook, its mask and its count, or fail.
    d.push(define(
        "zl_dbg_gethook_now",
        &[],
        any(),
        vec![
            when(
                and(
                    is_nil(read_global(HOOK_FN, any())),
                    eq(read_global(HOOK_MASK, i64()), int(0)),
                ),
                vec![ret(call("zl_fail", vec![], any()))],
            ),
            name.decl(text("")),
            when(
                has_event(HOOK_CALL),
                vec![name.set(add(name.e(), text("c")))],
            ),
            when(
                has_event(HOOK_RETURN),
                vec![name.set(add(name.e(), text("r")))],
            ),
            when(
                has_event(HOOK_LINE),
                vec![name.set(add(name.e(), text("l")))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        read_global(HOOK_FN, any()),
                        box_str(name.e()),
                        box_i64(read_global(HOOK_COUNT_EVERY, i64())),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    // `fail`: nil, as one value.
    d.push(define("zl_fail", &[], any(), vec![ret(nil())]));

    // ─── the rest ───────────────────────────────────────────────
    // A userdata's user values. The optional `n` is checked first.
    // The only full userdata is a foreign object, which has no user
    // values: a get answers fail, a set answers fail once it has its
    // value, and a set on anything else refuses its first argument.
    let opt_n = |number: usize, what: &str| {
        expr(call(
            "zl_arg_opt_int",
            vec![
                call("zl_value_at", vec![args.e(), int(number as i64)], any()),
                int(1),
                bad(number, what, ""),
            ],
            i64(),
        ))
    };
    d.push(define(
        "zl_debug_getuservalue",
        &[args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            opt_n(2, "getuservalue"),
            ret(call("zl_fail", vec![], any())),
        ],
    ));
    d.push(define(
        "zl_debug_setuservalue",
        &[args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            opt_n(3, "setuservalue"),
            x.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            when(
                zyntax_builtins::foreign::is_foreign(x.e()),
                vec![
                    when(
                        lt(len(args.e()), int(2)),
                        vec![lua_error(bad(2, "setuservalue", " (value expected)"))],
                    ),
                    ret(call("zl_fail", vec![], any())),
                ],
            ),
            lua_error(concat(vec![
                bad(1, "setuservalue", " (userdata expected, got "),
                if_expr(
                    lt(len(args.e()), int(1)),
                    text("no value"),
                    arg_type_name(x.e()),
                ),
                text(")"),
            ])),
            ret(call("zl_fail", vec![], any())),
        ],
    ));
    // `debug.debug()`: each line of the standard input run as a
    // chunk, until `cont` or the input's end; errors are reported and
    // the loop goes on.
    d.push(define(
        "zl_debug_debug",
        &[],
        unit(),
        vec![
            x.decl(nil()),
            y.decl(nil()),
            while_(
                bool(true),
                vec![
                    x.set(call("zl_io_read", vec![list(vec![], anys.clone())], any())),
                    when(
                        or(
                            is_nil(x.e()),
                            or(
                                ne(category(x.e()), int(STR)),
                                str_eq(get_str(x.e()), text("cont")),
                            ),
                        ),
                        vec![ret_void()],
                    ),
                    y.set(call(
                        "zl_load_raw",
                        vec![
                            get_str(x.e()),
                            text("=(debug command)"),
                            call("zl_globals_value", vec![], any()),
                        ],
                        any(),
                    )),
                    if_(
                        is_nil(y.e()),
                        vec![expr(call(
                            "zb_eprintln",
                            vec![call("zl_load_error", vec![], string())],
                            unit(),
                        ))],
                        vec![
                            expr(call("zl_call_0", vec![y.e()], any())),
                            when(
                                not(is_nil(pending())),
                                vec![expr(call(
                                    "zb_eprintln",
                                    vec![call(
                                        "zl_tostring",
                                        vec![call("zl_take_pending", vec![], any())],
                                        string(),
                                    )],
                                    unit(),
                                ))],
                            ),
                        ],
                    ),
                ],
            ),
            ret_void(),
        ],
    ));
    // The C stack's limit is not the program's to set.
    d.push(define(
        "zl_debug_setcstacklimit",
        &[args],
        i64(),
        vec![ret(int(0))],
    ));
}
