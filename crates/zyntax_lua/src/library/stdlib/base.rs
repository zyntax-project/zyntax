//! Base declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        acc,
        anys,
        args,
        i,
        j,
        k,
        n,
        out,
        table,
        tb,
        x,
        y,
        ..
    } = context;

    // ─── base ───────────────────────────────────────────────────
    d.push(define(
        "zl_print",
        &[args],
        unit(),
        vec![
            acc.decl(text("")),
            n.decl(len(args.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(gt(i.e(), int(0)), vec![acc.set(add(acc.e(), text("\t")))]),
                    acc.set(add(
                        acc.e(),
                        call("zl_tostring", vec![at(args.e(), i.e())], string()),
                    )),
                    i.add_assign(int(1)),
                ],
            ),
            // Written as bytes through the standard stream, as `io.write`
            // writes: a string is whatever bytes it holds.
            expr(call(
                "zl_stdout_write",
                vec![add(acc.e(), text("\n"))],
                i64(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_type",
        &[x],
        string(),
        vec![
            when(is_nil(x.e()), vec![ret(text("nil"))]),
            when(
                or(
                    eq(tag_of(x.e()), int(zyntax_builtins::FUNC_TAG)),
                    eq(tag_of(x.e()), int(zyntax_builtins::CODE_TAG)),
                ),
                vec![ret(text("function"))],
            ),
            when(
                or(is_file(x.e()), is_light(x.e())),
                vec![ret(text("userdata"))],
            ),
            ret(type_name(x.e())),
        ],
    ));
    d.push(define(
        "zl_tonumber_of",
        &[x, y],
        any(),
        vec![
            when(
                is_nil(y.e()),
                vec![ret(call("zl_tonumber", vec![x.e()], any()))],
            ),
            when(
                or(is_nil(x.e()), ne(category(x.e()), int(STR))),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'tonumber' (string expected, got "),
                    arg_type_name(x.e()),
                    text(")"),
                ]))],
            ),
            ret(call(
                "zl_tonumber_base",
                vec![
                    get_str(x.e()),
                    call("zl_arg_int", vec![y.e(), bad_arg(2, "tonumber")], i64()),
                ],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_next_of",
        &[tb, x],
        any(),
        vec![ret(call(
            "zb_box_tuple",
            vec![call("zl_next", vec![tb.e(), x.e()], anys.clone())],
            any(),
        ))],
    ));
    d.push(define(
        "zl_rawset_of",
        &[tb, x, y],
        table.clone(),
        vec![
            expr(call("zl_rawset", vec![tb.e(), x.e(), y.e()], unit())),
            ret(tb.e()),
        ],
    ));
    // `next` and `ipairs`'s iterator as function values, for `pairs`
    // and `ipairs` used as values rather than in a `for`.
    let env = borrowed("env", anys.clone());
    let a0 = local("a0", any());
    let a1 = local("a1", any());
    d.push(define(
        "zl_next_code",
        &[&env, &a0, &a1],
        any(),
        vec![ret(call(
            "zl_next_of",
            vec![
                call(
                    "zl_as_table",
                    vec![a0.e(), bad_arg(1, "next")],
                    table.clone(),
                ),
                a1.e(),
            ],
            any(),
        ))],
    ));
    d.push(define(
        "zl_ipairs_code",
        &[&env, &a0, &a1],
        any(),
        vec![
            i.decl(add(
                call(
                    "zl_arg_int",
                    vec![a1.e(), bad_arg(2, "for iterator")],
                    i64(),
                ),
                int(1),
            )),
            y.decl(call("zl_geti", vec![a0.e(), i.e()], any())),
            when(
                is_nil(y.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![], anys.clone())],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_i64(i.e()), y.e()], anys.clone())],
                any(),
            )),
        ],
    ));
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    // A builtin as a value is one record, whatever names it: the
    // reference compares functions by identity.
    d.push(global_var("zl_builtin_values", any()));
    let index = local("index", i64());
    let code = local("code", usize());
    let arity = local("arity", i64());
    d.push(define(
        "zl_builtin_value",
        &[&index, &code, &arity],
        any(),
        vec![
            when(
                is_nil(read_global("zl_builtin_values", any())),
                vec![set_global(
                    "zl_builtin_values",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            tb.decl(unbox_table(read_global("zl_builtin_values", any()), t)),
            y.decl(call("zl_rawgeti", vec![tb.e(), index.e()], any())),
            when(not(is_nil(y.e())), vec![ret(y.e())]),
            y.set(call("zl_func_of", vec![code.e(), arity.e()], any())),
            expr(call("zl_rawseti", vec![tb.e(), index.e(), y.e()], unit())),
            ret(y.e()),
        ],
    ));
    let builtin_value = |b: &Builtin| {
        let index = builtin_index(b);
        call(
            "zl_builtin_value",
            vec![
                int(index as i64),
                code_of(&wrapper_name(b)),
                int(VARIADIC_ARITY),
            ],
            any(),
        )
    };
    // The iterators `pairs` and `ipairs` give: `next` itself, and one
    // record for `ipairs`, past the builtins' slots.
    let next_value = || {
        let b = BUILTINS
            .iter()
            .find(|b| b.lib.is_empty() && b.name == "next")
            .expect("next");
        builtin_value(b)
    };
    let ipairs_value = || {
        call(
            "zl_builtin_value",
            vec![
                int(BUILTINS.len() as i64),
                code_of("zl_ipairs_code"),
                int(2),
            ],
            any(),
        )
    };
    d.push(define(
        "zl_pairs",
        &[x],
        any(),
        vec![
            y.decl(call("zl_meta_of", vec![x.e(), text("__pairs")], any())),
            when(
                not(is_nil(y.e())),
                vec![ret(call("zl_call_1", vec![y.e(), x.e()], any()))],
            ),
            when(
                not(is_table(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'for iterator' (table expected, got "),
                    arg_type_name(x.e()),
                    text(")"),
                ]))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![next_value(), x.e(), nil()], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_ipairs",
        &[x],
        any(),
        vec![ret(call(
            "zb_box_tuple",
            vec![list(
                vec![ipairs_value(), x.e(), box_i64(int(0))],
                anys.clone(),
            )],
            any(),
        ))],
    ));
    d.push(define(
        "zl_select",
        &[x, args],
        any(),
        vec![
            n.decl(len(args.e())),
            when(
                and(
                    eq(category(x.e()), int(STR)),
                    str_eq(get_str(x.e()), text("#")),
                ),
                vec![ret(box_i64(n.e()))],
            ),
            i.decl(call("zl_arg_int", vec![x.e(), bad_arg(1, "select")], i64())),
            when(
                lt(i.e(), int(0)),
                vec![i.set(add(add(n.e(), i.e()), int(1)))],
            ),
            when(
                lt(i.e(), int(1)),
                vec![lua_error(text(
                    "bad argument #1 to 'select' (index out of range)",
                ))],
            ),
            ret(call("zl_values_from", vec![args.e(), i.e()], any())),
        ],
    ));
    // Equality without metamethods: the same value.
    d.push(define(
        "zl_rawequal",
        &[x, y],
        boolean(),
        vec![
            when(
                and(is_table(x.e()), is_table(y.e())),
                vec![ret(eq(
                    call("zb_unbox_instance_raw", vec![x.e()], i64()),
                    call("zb_unbox_instance_raw", vec![y.e()], i64()),
                ))],
            ),
            ret(call("zl_eq", vec![x.e(), y.e()], boolean())),
        ],
    ));
    d.push(define(
        "zl_rawlen",
        &[x],
        i64(),
        vec![
            when(
                is_table(x.e()),
                vec![ret(call("zl_len", vec![unbox_table(x.e(), t)], i64()))],
            ),
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(STR))),
                vec![ret(call("zb_str_len", vec![get_str(x.e())], i64()))],
            ),
            lua_error(concat(vec![
                text("bad argument #1 to 'rawlen' (table or string expected, got "),
                arg_type_name(x.e()),
                text(")"),
            ])),
            ret(int(0)),
        ],
    ));
    // `assert(v, message)`: the message is raised as `error` raises
    // it, a string positioned.
    d.push(define(
        "zl_assert",
        &[x, args],
        any(),
        vec![
            when(
                not(call("zl_truthy", vec![x.e()], boolean())),
                vec![
                    if_(
                        ge(len(args.e()), int(1)),
                        vec![expr(call(
                            "zl_error",
                            vec![at(args.e(), int(0)), int(1)],
                            unit(),
                        ))],
                        vec![expr(call(
                            "zl_raise_value",
                            vec![box_str(call(
                                "zl_position",
                                vec![text("assertion failed!")],
                                string(),
                            ))],
                            unit(),
                        ))],
                    ),
                    ret(nil()),
                ],
            ),
            expr(mcall(args.e(), "insert_at", vec![int(0), x.e()], unit())),
            ret(call("zl_pack", vec![args.e()], any())),
        ],
    ));
    // `pcall(f, ...)`: true and the results, or false and the error.
    // The call is from no line: what it calls directly reports no
    // position, as under the reference where the caller is C.
    let handler = kept("handler", any());
    let err = kept("err", any());
    // Past the C stack's limit, the call is refused in pcall's own
    // frame, so the message has no position.
    let c_stack_full = || {
        when(
            ge(read_global(CCALLS, i64()), int(CCALLS_LIMIT)),
            vec![ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![box_bool(bool(false)), box_str(text("C stack overflow"))],
                    anys.clone(),
                )],
                any(),
            ))],
        )
    };
    d.push(define(
        "zl_pcall",
        &[x, args],
        any(),
        vec![
            c_stack_full(),
            set_global(LINE, int(0)),
            y.decl(call("zl_call_packed", vec![x.e(), args.e()], any())),
            // A coroutine being closed unwinds through the call.
            when(is_closing(pending()), vec![ret(nil())]),
            when(
                not(is_nil(pending())),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            box_bool(bool(false)),
                            call("zl_take_pending", vec![], any()),
                        ],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            out.decl(list(vec![box_bool(bool(true))], anys.clone())),
            expr(call("zl_append_values", vec![out.e(), y.e()], unit())),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    // `xpcall(f, handler, ...)`: the handler sees the error first. A
    // handler that raises is called again on its own error, some
    // times; when it overflows the stack, or past them, the error is
    // an error in error handling.
    let failed = |err: Expr| {
        ret(call(
            "zb_box_tuple",
            vec![list(vec![box_bool(bool(false)), err], anys.clone())],
            any(),
        ))
    };
    let hdepth = local("hdepth", i64());
    let overflow = local("overflow", boolean());
    let handling = local("handling", boolean());
    d.push(define(
        "zl_xpcall",
        &[x, &handler, args],
        any(),
        vec![
            when(
                not(is_func(handler.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #2 to 'xpcall' (function expected, got "),
                    arg_type_name(handler.e()),
                    text(")"),
                ]))],
            ),
            c_stack_full(),
            set_global(LINE, int(0)),
            y.decl(call("zl_call_packed", vec![x.e(), args.e()], any())),
            when(is_closing(pending()), vec![ret(nil())]),
            when(
                not(is_nil(pending())),
                vec![
                    k.decl(int(0)),
                    overflow.decl(read_global(OVERFLOWED, boolean())),
                    handling.decl(read_global(HANDLING_OVERFLOW, boolean())),
                    err.decl(call("zl_take_pending", vec![], any())),
                    while_(
                        bool(true),
                        vec![
                            // In a program that keeps its call stack,
                            // the handler runs on the stack the error
                            // was raised on.
                            hdepth.decl(int(-1)),
                            when(
                                read_global(debug::DBG_ON, boolean()),
                                vec![
                                    hdepth.set(call("zl_dbg_handler_enter", vec![], i64())),
                                    when(
                                        ge(hdepth.e(), int(0)),
                                        vec![set_global(
                                            LINE,
                                            call("zl_dbg_raise_line", vec![], i64()),
                                        )],
                                    ),
                                    set_global(debug::DBG_SITE, int(0)),
                                ],
                            ),
                            set_global(HANDLING_OVERFLOW, or(handling.e(), overflow.e())),
                            err.set(call(
                                "zl_first",
                                vec![call("zl_call_1", vec![handler.e(), err.e()], any())],
                                any(),
                            )),
                            set_global(HANDLING_OVERFLOW, handling.e()),
                            when(
                                ge(hdepth.e(), int(0)),
                                vec![
                                    expr(call("zl_dbg_handler_leave", vec![hdepth.e()], unit())),
                                    set_global(LINE, int(0)),
                                ],
                            ),
                            when(is_nil(pending()), vec![failed(err.e())]),
                            when(
                                or(
                                    read_global(OVERFLOWED, boolean()),
                                    ge(k.e(), int(HANDLER_RETRIES)),
                                ),
                                vec![
                                    expr(call("zl_take_pending", vec![], any())),
                                    failed(box_str(text("error in error handling"))),
                                ],
                            ),
                            err.set(call("zl_take_pending", vec![], any())),
                            k.add_assign(int(1)),
                        ],
                    ),
                ],
            ),
            out.decl(list(vec![box_bool(bool(true))], anys.clone())),
            expr(call("zl_append_values", vec![out.e(), y.e()], unit())),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    // `table.unpack(t, i, j)`: the values `t[i]..t[j]`, `j` the
    // length when not given; any integer may be given.
    let ok = local("ok", boolean());
    let last = kept("last", any());
    d.push(define(
        "zl_unpack",
        &[tb, i, &last],
        any(),
        vec![
            j.decl(if_expr(
                is_nil(last.e()),
                call("zl_table_len", vec![tb.e()], i64()),
                call("zl_arg_int", vec![last.e(), bad_arg(3, "unpack")], i64()),
            )),
            // The reference's stack holds a million values, of which
            // the caller's frame and its minimum reserve take twenty:
            // `j - i + 1 + 20 > 1_000_000` is refused.
            when(
                and(
                    le(i.e(), j.e()),
                    or(
                        ge(sub(cast(j.e(), u64()), cast(i.e(), u64())), int(999_980)),
                        lt(sub(j.e(), i.e()), int(0)),
                    ),
                ),
                vec![lua_error(text("too many results to unpack"))],
            ),
            out.decl(list(vec![], anys.clone())),
            // The loop stops on reaching `j` rather than passing it, so
            // an index at the end of the integers is fine.
            k.decl(i.e()),
            ok.decl(le(i.e(), j.e())),
            while_(
                ok.e(),
                vec![
                    push(out.e(), call("zl_table_geti", vec![tb.e(), k.e()], any())),
                    if_(
                        eq(k.e(), j.e()),
                        vec![ok.set(bool(false))],
                        vec![k.add_assign(int(1))],
                    ),
                ],
            ),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
}
