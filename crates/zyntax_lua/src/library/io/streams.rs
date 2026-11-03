//! Streams declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        a,
        anys,
        args,
        fmt,
        h,
        i,
        k,
        n,
        ok,
        out,
        r,
        s,
        x,
        y,
        ..
    } = context;

    // ─── files as values ────────────────────────────────────────
    let handle_of = |x: Expr| call("zb_unbox_instance_raw", vec![x], i64());
    let is_open = |x: Expr| call("zl_io_is_open", vec![handle_of(x)], boolean());
    // What a file operation that failed answers: nil, the message and
    // the error's number.
    let failure = || {
        ret(call(
            "zb_box_tuple",
            vec![list(
                vec![
                    nil(),
                    box_str(call("zl_io_error", vec![], string())),
                    box_i64(call("zl_io_errno", vec![], i64())),
                ],
                anys.clone(),
            )],
            any(),
        ))
    };
    let one = |v: Expr| {
        ret(call(
            "zb_box_tuple",
            vec![list(vec![v], anys.clone())],
            any(),
        ))
    };

    // ─── reading and writing ────────────────────────────────────
    // `f:read(...)`: a value per format, the first that fails nil and
    // the last; the line without its end when no format is given.
    let read_line = |keep: bool| s.decl(call("zl_io_read_line", vec![h.e(), bool(keep)], string()));
    let string_or_nil = || if_expr(eq(s.e(), null(string())), nil(), box_str(s.e()));
    d.push(define(
        "zl_file_read",
        &[x, args],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "read")],
                i64(),
            )),
            expr(call("zl_io_failed", vec![], boolean())),
            out.decl(list(vec![], anys.clone())),
            n.decl(len(args.e())),
            when(
                eq(n.e(), int(0)),
                vec![
                    read_line(false),
                    when(call("zl_io_failed", vec![], boolean()), vec![failure()]),
                    y.decl(string_or_nil()),
                    push(out.e(), y.e()),
                    ret(call("zb_box_tuple", vec![out.e()], any())),
                ],
            ),
            i.decl(int(0)),
            ok.decl(bool(true)),
            while_(
                and(lt(i.e(), n.e()), ok.e()),
                vec![
                    a.decl(at(args.e(), i.e())),
                    y.decl(nil()),
                    if_(
                        and(
                            not(is_nil(a.e())),
                            or(
                                eq(category(a.e()), int(INT)),
                                or(
                                    eq(category(a.e()), int(UINT)),
                                    eq(category(a.e()), int(FLOAT)),
                                ),
                            ),
                        ),
                        vec![
                            k.decl(call(
                                "zl_arg_int",
                                vec![a.e(), bad_arg_at(i.e(), "read")],
                                i64(),
                            )),
                            s.decl(call("zl_io_read_bytes", vec![h.e(), k.e()], string())),
                            y.set(string_or_nil()),
                        ],
                        vec![
                            when(
                                or(is_nil(a.e()), ne(category(a.e()), int(STR))),
                                vec![lua_error(concat(vec![
                                    bad_arg_at(i.e(), "read"),
                                    text(" (string expected, got "),
                                    arg_type_name(a.e()),
                                    text(")"),
                                ]))],
                            ),
                            fmt.decl(get_str(a.e())),
                            when(
                                call("zb_str_startswith", vec![fmt.e(), text("*")], boolean()),
                                vec![fmt.set(call(
                                    "zb_str_substring",
                                    vec![fmt.e(), int(1), call("zb_str_len", vec![fmt.e()], i64())],
                                    string(),
                                ))],
                            ),
                            k.decl(if_expr(
                                eq(call("zb_str_len", vec![fmt.e()], i64()), int(0)),
                                int(0),
                                cast(call("zb_str_code_at", vec![fmt.e(), int(0)], i32()), i64()),
                            )),
                            if_(
                                eq(k.e(), int(b'n' as i64)),
                                vec![
                                    r.decl(call("zl_io_read_number", vec![h.e()], i64())),
                                    when(
                                        eq(r.e(), int(1)),
                                        vec![y.set(box_i64(call(
                                            "zl_io_number_int",
                                            vec![],
                                            i64(),
                                        )))],
                                    ),
                                    when(
                                        eq(r.e(), int(2)),
                                        vec![y.set(box_f64(call(
                                            "zl_io_number_float",
                                            vec![],
                                            f64(),
                                        )))],
                                    ),
                                ],
                                vec![if_(
                                    eq(k.e(), int(b'l' as i64)),
                                    vec![read_line(false), y.set(string_or_nil())],
                                    vec![if_(
                                        eq(k.e(), int(b'L' as i64)),
                                        vec![read_line(true), y.set(string_or_nil())],
                                        vec![if_(
                                            eq(k.e(), int(b'a' as i64)),
                                            vec![
                                                s.decl(call(
                                                    "zl_io_read_all",
                                                    vec![h.e()],
                                                    string(),
                                                )),
                                                y.set(if_expr(
                                                    eq(s.e(), null(string())),
                                                    box_str(text("")),
                                                    box_str(s.e()),
                                                )),
                                            ],
                                            vec![lua_error(add(
                                                bad_arg_at(i.e(), "read"),
                                                text(" (invalid format)"),
                                            ))],
                                        )],
                                    )],
                                )],
                            ),
                        ],
                    ),
                    when(call("zl_io_failed", vec![], boolean()), vec![failure()]),
                    push(out.e(), y.e()),
                    when(is_nil(y.e()), vec![ok.set(bool(false))]),
                    i.add_assign(int(1)),
                ],
            ),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));
    // `f:write(...)`: strings and numbers, the file back.
    d.push(define(
        "zl_file_write",
        &[x, args],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "write")],
                i64(),
            )),
            i.decl(int(0)),
            n.decl(len(args.e())),
            while_(
                lt(i.e(), n.e()),
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
                            bad_arg_at(i.e(), "write"),
                            text(" (string expected, got "),
                            arg_type_name(a.e()),
                            text(")"),
                        ]))],
                    ),
                    s.decl(if_expr(
                        eq(category(a.e()), int(FLOAT)),
                        call("zl_format_g", vec![get_f64(a.e()), int(14)], string()),
                        call("zl_arg_str", vec![a.e(), text("")], string()),
                    )),
                    r.decl(call("zl_io_write", vec![h.e(), s.e()], i64())),
                    when(ne(r.e(), int(0)), vec![failure()]),
                    i.add_assign(int(1)),
                ],
            ),
            one(x.e()),
        ],
    ));
    // `f:close()`: true, or a command's outcome as `os.execute` gives
    // it; a standard stream stays open.
    d.push(define(
        "zl_file_close",
        &[x],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "close")],
                i64(),
            )),
            when(
                le(h.e(), int(STDERR)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(text("cannot close standard file"))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ok.decl(call("zl_io_is_pipe", vec![h.e()], boolean())),
            r.decl(call("zl_io_close", vec![h.e()], i64())),
            when(
                ok.e(),
                vec![
                    when(eq(bitand(r.e(), int(1)), int(0)), vec![failure()]),
                    r.set(shr(r.e(), int(1))),
                    k.decl(call("zl_exec_result", vec![r.e(), bool(false)], i64())),
                    y.decl(nil()),
                    when(
                        and(
                            eq(
                                call("zl_exec_result", vec![r.e(), bool(true)], i64()),
                                int(0),
                            ),
                            eq(k.e(), int(0)),
                        ),
                        vec![y.set(box_bool(bool(true)))],
                    ),
                    a.decl(box_str(if_expr(
                        eq(
                            call("zl_exec_result", vec![r.e(), bool(true)], i64()),
                            int(0),
                        ),
                        text("exit"),
                        text("signal"),
                    ))),
                    ret(call(
                        "zb_box_tuple",
                        vec![list(vec![y.e(), a.e(), box_i64(k.e())], anys.clone())],
                        any(),
                    )),
                ],
            ),
            when(ne(r.e(), int(0)), vec![failure()]),
            one(box_bool(bool(true))),
        ],
    ));
    // `f:seek([whence [, offset]])`.
    d.push(define(
        "zl_file_seek",
        &[x, y, a],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "seek")],
                i64(),
            )),
            s.decl(if_expr(
                is_nil(y.e()),
                text("cur"),
                call("zl_arg_str", vec![y.e(), bad_arg(1, "seek")], string()),
            )),
            k.decl(int(-1)),
            when(str_eq(s.e(), text("set")), vec![k.set(int(0))]),
            when(str_eq(s.e(), text("cur")), vec![k.set(int(1))]),
            when(str_eq(s.e(), text("end")), vec![k.set(int(2))]),
            when(
                lt(k.e(), int(0)),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'seek' (invalid option '"),
                    s.e(),
                    text("')"),
                ]))],
            ),
            n.decl(if_expr(
                is_nil(a.e()),
                int(0),
                call("zl_arg_int", vec![a.e(), bad_arg(2, "seek")], i64()),
            )),
            r.decl(call("zl_io_seek", vec![h.e(), k.e(), n.e()], i64())),
            when(lt(r.e(), int(0)), vec![failure()]),
            one(box_i64(r.e())),
        ],
    ));
    d.push(define(
        "zl_file_flush",
        &[x],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "flush")],
                i64(),
            )),
            r.decl(call("zl_io_flush", vec![h.e()], i64())),
            when(ne(r.e(), int(0)), vec![failure()]),
            one(box_bool(bool(true))),
        ],
    ));
    // `f:setvbuf(mode [, size])`.
    d.push(define(
        "zl_file_setvbuf",
        &[x, y, a],
        any(),
        vec![
            h.decl(call(
                "zl_file_check",
                vec![x.e(), bad_arg(1, "setvbuf")],
                i64(),
            )),
            s.decl(call(
                "zl_arg_str",
                vec![y.e(), bad_arg(1, "setvbuf")],
                string(),
            )),
            when(
                not(or(
                    str_eq(s.e(), text("no")),
                    or(str_eq(s.e(), text("full")), str_eq(s.e(), text("line"))),
                )),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'setvbuf' (invalid option '"),
                    s.e(),
                    text("')"),
                ]))],
            ),
            n.decl(if_expr(
                is_nil(a.e()),
                int(8192),
                call("zl_arg_int", vec![a.e(), bad_arg(2, "setvbuf")], i64()),
            )),
            k.decl(int(2)),
            when(str_eq(s.e(), text("no")), vec![k.set(int(0))]),
            when(str_eq(s.e(), text("line")), vec![k.set(int(1))]),
            r.decl(call("zl_io_setvbuf", vec![h.e(), k.e(), n.e()], i64())),
            when(ne(r.e(), int(0)), vec![failure()]),
            one(box_bool(bool(true))),
        ],
    ));
    // `tostring(f)`.
    d.push(define(
        "zl_file_str",
        &[x],
        string(),
        vec![
            when(not(is_open(x.e())), vec![ret(text("file (closed)"))]),
            ret(concat(vec![
                text("file (0x"),
                call(
                    "zb_str_of_int_radix",
                    vec![add(int(0x7f00), mul(handle_of(x.e()), int(16))), int32(16)],
                    string(),
                ),
                text(")"),
            ])),
        ],
    ));
    // `__gc` and `__close`: an open file is closed; a standard stream
    // is left alone.
    d.push(define(
        "zl_file_release",
        &[x],
        any(),
        vec![
            when(
                and(is_open(x.e()), gt(handle_of(x.e()), int(STDERR))),
                vec![expr(call("zl_io_close", vec![handle_of(x.e())], i64()))],
            ),
            ret(call("zl_none", vec![], any())),
        ],
    ));
}
