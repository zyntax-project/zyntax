//! Packing declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        i,
        j,
        k,
        n,
        out,
        s,
        x,
        y,
        ..
    } = context;
    let is_int_box = |x: Expr| {
        let c = category(x);
        or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
    };

    // ─── string.pack ────────────────────────────────────────────
    let packed = kept("packed", string());
    d.push(define(
        "zl_string_pack",
        &[s, args],
        string(),
        vec![
            packed.decl(call("zl_pack_raw", vec![s.e(), args.e()], string())),
            when(
                eq(packed.e(), null(string())),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(packed.e()),
        ],
    ));
    d.push(define(
        "zl_string_packsize",
        &[s],
        i64(),
        vec![
            n.decl(call("zl_packsize_raw", vec![s.e()], i64())),
            when(
                lt(n.e(), int(0)),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(n.e()),
        ],
    ));
    // `string.unpack(fmt, s, pos)`: the position is Lua's, relative
    // from the end when negative; the values come back one by one.
    let data = kept("data", string());
    let data_any = kept("data", any());
    let pos = local("pos", i64());
    for (name, data, len_of, raw) in [
        ("zl_string_unpack", &data, "zb_str_len", "zl_unpack_raw"),
        (
            "zl_string_unpack$any",
            &data_any,
            "zl_string_len$any",
            "zl_unpack_raw$any",
        ),
    ] {
        d.push(define(
            name,
            &[s, data, &pos],
            any(),
            vec![
                n.decl(call(len_of, vec![data.e()], i64())),
                when(
                    lt(pos.e(), int(0)),
                    vec![pos.set(if_expr(
                        lt(pos.e(), sub(int(0), n.e())),
                        int(1),
                        add(add(n.e(), pos.e()), int(1)),
                    ))],
                ),
                when(eq(pos.e(), int(0)), vec![pos.set(int(1))]),
                k.decl(call(
                    raw,
                    vec![s.e(), data.e(), sub(pos.e(), int(1))],
                    i64(),
                )),
                when(
                    lt(k.e(), int(0)),
                    vec![lua_error(call("zl_pack_error", vec![], string()))],
                ),
                out.decl(list(vec![], anys.clone())),
                i.decl(int(0)),
                while_(
                    lt(i.e(), k.e()),
                    vec![
                        j.decl(call("zl_unpack_kind", vec![i.e()], i64())),
                        when(
                            eq(j.e(), int(0)),
                            vec![push(
                                out.e(),
                                box_i64(call("zl_unpack_int", vec![i.e()], i64())),
                            )],
                        ),
                        when(
                            eq(j.e(), int(1)),
                            vec![push(
                                out.e(),
                                box_f64(call("zl_unpack_float", vec![i.e()], f64())),
                            )],
                        ),
                        when(
                            eq(j.e(), int(2)),
                            vec![push(
                                out.e(),
                                box_str(call("zl_unpack_str", vec![i.e()], string())),
                            )],
                        ),
                        i.add_assign(int(1)),
                    ],
                ),
                push(out.e(), box_i64(call("zl_unpack_next", vec![], i64()))),
                ret(call("zb_box_tuple", vec![out.e()], any())),
            ],
        ));
    }

    // ─── os and io ──────────────────────────────────────────────
    d.extend(os_declarations(t));
    // `os.exit(code, close)`: with `close` true, the state is closed
    // first: an error nothing catches unwinds the main thread, closing
    // its variables, and the host runs the finalizers and exits once it
    // is out.
    d.push(define(
        "zl_os_exit",
        &[x, y],
        unit(),
        vec![
            k.decl(int(0)),
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(BOOL))),
                vec![when(not(get_bool(x.e())), vec![k.set(int(1))])],
            ),
            when(
                and(not(is_nil(x.e())), is_int_box(x.e())),
                vec![k.set(get_i64(x.e()))],
            ),
            when(
                call("zl_truthy", vec![y.e()], boolean()),
                vec![
                    expr(call("zl_raise_value", vec![exit_marker(k.e())], unit())),
                    ret_void(),
                ],
            ),
            // `exit` flushes C's streams, not the program's own.
            expr(call("zl_io_flush_all", vec![], unit())),
            expr(call("zb_exit", vec![cast(k.e(), i32())], unit())),
            ret_void(),
        ],
    ));
    // A float as `%.14g`, without the `.0` that `tostring` adds, for
    // `io.write`.
    d.push(extern_fn(
        "zl_format_g",
        &[("x", f64()), ("precision", i64())],
        string(),
        Some("$Lua$format_g"),
    ));
}
