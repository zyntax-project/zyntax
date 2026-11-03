//! Locals declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        first,
        i,
        j,
        k,
        level,
        n,
        name,
        thread,
        x,
        y,
        z,
        ..
    } = context;
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
    let int_arg = |x: Expr, what: &str, number: usize| {
        call("zl_arg_int", vec![x, bad(number, what, "")], i64())
    };
    let pair = |a: Expr, b: Expr| call("zb_box_tuple", vec![list(vec![a, b], anys.clone())], any());

    // ─── locals ─────────────────────────────────────────────────
    // Local `n` of stack level `level`: where its value lives in the
    // list the frame spilled its locals to (a list of the values, the
    // extra arguments of a variadic function boxed after them), its
    // name, and the list; an empty name when there is no such local.
    // `n` below zero counts the extra arguments.
    let depth = local("depth", i64());
    let spill = borrowed("spill", anys.clone());
    let lookup = |who: &str, st: &mut Vec<Stmt>| {
        st.extend([
            depth.decl(call("zl_dbg_frame_depth", vec![level.e()], i64())),
            when(
                lt(depth.e(), int(0)),
                vec![lua_error(bad(1, who, " (level out of range)"))],
            ),
            name.decl(text("")),
            x.decl(nil()),
            k.decl(int(0)),
            when(
                gt(depth.e(), int(0)),
                vec![
                    k.set(call("zl_dbg_frame_spill", vec![depth.e()], i64())),
                    when(
                        ne(k.e(), int(0)),
                        vec![x.set(call("zl_dbg_word_as_any", vec![k.e()], any()))],
                    ),
                ],
            ),
        ]);
    };
    // Reads `name` and the slot `i` of `spill` the local is at, or
    // leaves `name` empty.
    let find = |st: &mut Vec<Stmt>| {
        st.push(when(
            not(is_nil(x.e())),
            vec![
                spill.decl(call("zb_unbox_list_raw_any", vec![x.e()], anys.clone())),
                j.decl(call("zl_dbg_local_count", vec![depth.e()], i64())),
                if_(
                    gt(n.e(), int(0)),
                    vec![when(
                        and(le(n.e(), j.e()), le(n.e(), len(spill.e()))),
                        vec![
                            name.set(call("zl_dbg_local_name", vec![depth.e(), n.e()], string())),
                            i.set(sub(n.e(), int(1))),
                        ],
                    )],
                    vec![when(
                        and(lt(n.e(), int(0)), gt(len(spill.e()), j.e())),
                        vec![
                            y.set(at(spill.e(), j.e())),
                            z.set(call(
                                "zl_value_at",
                                vec![call("zl_values", vec![y.e()], anys.clone()), neg(n.e())],
                                any(),
                            )),
                            when(
                                le(
                                    neg(n.e()),
                                    len(call("zl_values", vec![y.e()], anys.clone())),
                                ),
                                vec![name.set(text("(vararg)")), i.set(int(-1))],
                            ),
                        ],
                    )],
                ),
            ],
        ));
    };

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(args, thread, first));
    st.extend([
        y.decl(arg(int(1))),
        z.decl(nil()),
        when(
            lt(len(args.e()), add(first.e(), int(2))),
            vec![lua_error(bad(
                2,
                "getlocal",
                " (number expected, got no value)",
            ))],
        ),
        n.decl(int_arg(arg(int(2)), "getlocal", 2)),
        i.decl(int(0)),
        j.decl(int(0)),
        // A function: the name of its parameter `n`.
        when(
            is_func(y.e()),
            vec![
                name.decl(call(
                    "zl_dbg_param_name",
                    vec![call("zl_func_id", vec![y.e()], i64()), n.e()],
                    string(),
                )),
                when(
                    eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                    vec![ret(call("zl_fail", vec![], any()))],
                ),
                ret(box_str(name.e())),
            ],
        ),
        // Another thread's frames keep no locals here.
        when(
            ne(thread.e(), int(-1)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        level.decl(call(
            "zl_arg_int",
            vec![y.e(), bad(1, "getlocal", "")],
            i64(),
        )),
    ]);
    lookup("getlocal", &mut st);
    find(&mut st);
    st.extend([
        when(
            eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        when(
            eq(i.e(), int(-1)),
            vec![ret(pair(box_str(name.e()), z.e()))],
        ),
        y.set(at(
            call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
            i.e(),
        )),
        // A local kept in a cell was spilled as the cell.
        when(
            ne(
                call("zl_dbg_local_cell", vec![depth.e(), n.e()], i64()),
                int(0),
            ),
            vec![y.set(at(
                call("zb_unbox_list_raw_any", vec![y.e()], anys.clone()),
                int(0),
            ))],
        ),
        ret(pair(box_str(name.e()), y.e())),
    ]);
    d.push(define("zl_debug_getlocal", &[args], any(), st));

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(args, thread, first));
    st.extend([
        y.decl(nil()),
        z.decl(nil()),
        level.decl(int_arg(arg(int(1)), "setlocal", 1)),
        n.decl(int_arg(arg(int(2)), "setlocal", 2)),
        when(
            lt(len(args.e()), add(first.e(), int(3))),
            vec![lua_error(bad(3, "setlocal", " (value expected)"))],
        ),
        when(
            ne(thread.e(), int(-1)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        i.decl(int(0)),
        j.decl(int(0)),
    ]);
    lookup("setlocal", &mut st);
    find(&mut st);
    st.extend([
        when(
            or(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                lt(i.e(), int(0)),
            ),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        // A local kept in a cell is set through it, where every closure
        // sharing it sees the value; any other is read back by its
        // frame when the call it stopped at returns.
        if_(
            ne(
                call("zl_dbg_local_cell", vec![depth.e(), n.e()], i64()),
                int(0),
            ),
            vec![set_idx(
                call(
                    "zb_unbox_list_raw_any",
                    vec![at(
                        call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                        i.e(),
                    )],
                    anys.clone(),
                ),
                int(0),
                arg(int(3)),
            )],
            vec![
                set_idx(
                    call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                    i.e(),
                    arg(int(3)),
                ),
                expr(call("zl_dbg_mark_set", vec![depth.e(), i.e()], unit())),
            ],
        ),
        ret(box_str(name.e())),
    ]);
    d.push(define("zl_debug_setlocal", &[args], any(), st));
}
