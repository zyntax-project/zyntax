//! Upvalues declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        acc,
        anys,
        args,
        f,
        key,
        n,
        name,
        x,
        ..
    } = context;

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

    // ─── upvalues ───────────────────────────────────────────────
    // The accessor of the chunk a function of the program belongs to.
    d.push(define(
        "zl_dbg_accessor",
        &[key],
        any(),
        vec![
            when(
                or(lt(key.e(), int(0)), is_nil(read_global(UPS, any()))),
                vec![ret(nil())],
            ),
            ret(call(
                "zl_rawgeti",
                vec![
                    unbox_table(read_global(UPS, any()), t),
                    shr(key.e(), int(32)),
                ],
                any(),
            )),
        ],
    ));
    // Upvalue `n` of `f` asked `how`: what the chunk's accessor answers.
    let how = local("how", i64());
    d.push(define(
        "zl_dbg_upvalue",
        &[f, n, x, &how],
        any(),
        vec![
            key.decl(call("zl_func_id", vec![f.e()], i64())),
            acc.decl(call("zl_dbg_accessor", vec![key.e()], any())),
            when(is_nil(acc.e()), vec![ret(nil())]),
            ret(call(
                "zl_apply_5",
                vec![
                    acc.e(),
                    box_i64(key.e()),
                    f.e(),
                    box_i64(n.e()),
                    x.e(),
                    box_i64(how.e()),
                ],
                any(),
            )),
        ],
    ));
    // The name of upvalue `n` of `f`: empty when it has none.
    d.push(define(
        "zl_dbg_upvalue_name_of",
        &[f, n],
        string(),
        vec![
            key.decl(call("zl_func_id", vec![f.e()], i64())),
            when(lt(key.e(), int(0)), vec![ret(text(""))]),
            ret(call("zl_dbg_upvalue_name", vec![key.e(), n.e()], string())),
        ],
    ));
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
    let none = || call("zl_none", vec![], any());
    let pair = |a: Expr, b: Expr| call("zb_box_tuple", vec![list(vec![a, b], anys.clone())], any());
    // `debug.getupvalue(f, n)`: the name and value, or nothing.
    d.push(define(
        "zl_debug_getupvalue",
        &[args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(f, "getupvalue", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "getupvalue",
                2,
            )),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(none())],
            ),
            ret(pair(
                box_str(name.e()),
                call(
                    "zl_dbg_upvalue",
                    vec![f.e(), n.e(), nil(), int(UP_GET)],
                    any(),
                ),
            )),
        ],
    ));
    // `debug.setupvalue(f, n, v)`: the name, or nothing.
    d.push(define(
        "zl_debug_setupvalue",
        &[args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(f, "setupvalue", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "setupvalue",
                2,
            )),
            when(
                lt(len(args.e()), int(3)),
                vec![lua_error(bad(3, "setupvalue", " (value expected)"))],
            ),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(none())],
            ),
            expr(call(
                "zl_dbg_upvalue",
                vec![
                    f.e(),
                    n.e(),
                    call("zl_value_at", vec![args.e(), int(3)], any()),
                    int(UP_SET),
                ],
                any(),
            )),
            ret(box_str(name.e())),
        ],
    ));
    // `debug.upvalueid(f, n)`: what tells upvalues apart.
    d.push(define(
        "zl_debug_upvalueid",
        &[args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(f, "upvalueid", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "upvalueid",
                2,
            )),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(call("zl_fail", vec![], any()))],
            ),
            ret(call(
                "zl_dbg_upvalue",
                vec![f.e(), n.e(), nil(), int(UP_ID)],
                any(),
            )),
        ],
    ));
    // `debug.upvaluejoin(f1, n1, f2, n2)`: upvalue `n1` of `f1` is
    // upvalue `n2` of `f2` from now on.
    let check_index = |f: Expr, n: Expr, number: usize| {
        when(
            eq(
                call(
                    "zb_str_len",
                    vec![call("zl_dbg_upvalue_name_of", vec![f, n], string())],
                    i64(),
                ),
                int(0),
            ),
            vec![lua_error(bad(
                number,
                "upvaluejoin",
                " (invalid upvalue index)",
            ))],
        )
    };
    let lua_function = |f: Expr, number: usize| {
        when(
            lt(call("zl_func_id", vec![f], i64()), int(0)),
            vec![lua_error(bad(
                number,
                "upvaluejoin",
                " (Lua function expected)",
            ))],
        )
    };
    let g = kept("g", any());
    let m = local("m", i64());
    d.push(define(
        "zl_debug_upvaluejoin",
        &[args],
        unit(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(f, "upvaluejoin", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "upvaluejoin",
                2,
            )),
            g.decl(call("zl_value_at", vec![args.e(), int(3)], any())),
            function_arg(&g, "upvaluejoin", 3),
            m.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(4)], any()),
                "upvaluejoin",
                4,
            )),
            check_index(f.e(), n.e(), 2),
            check_index(g.e(), m.e(), 4),
            lua_function(f.e(), 1),
            lua_function(g.e(), 3),
            x.decl(call(
                "zl_dbg_upvalue",
                vec![g.e(), m.e(), nil(), int(UP_CELL)],
                any(),
            )),
            expr(call(
                "zl_dbg_upvalue",
                vec![f.e(), n.e(), x.e(), int(UP_JOIN)],
                any(),
            )),
            ret_void(),
        ],
    ));
}
