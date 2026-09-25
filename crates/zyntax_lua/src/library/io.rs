//! The `io` library: files as values of their own kind over the host's
//! streams, the default input and output, and the reference's file
//! methods reached through the file metatable.

use super::*;

/// The last of the standard streams' handles, as the host numbers
/// them: stdin, stdout and stderr are 1, 2 and 3.
const STDERR: i64 = 3;

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let x = kept("x", any());
    let y = kept("y", any());
    let a = kept("a", any());
    let args = kept("args", anys.clone());
    let out = borrowed("out", anys.clone());
    let vals = borrowed("vals", anys.clone());
    let env = borrowed("env", anys.clone());
    let packed = kept("packed", any());
    let what = kept("what", string());
    let name = kept("name", string());
    let mode = kept("mode", string());
    let s = kept("s", string());
    let fmt = kept("fmt", string());
    let h = local("h", i64());
    let i = local("i", i64());
    let n = local("n", i64());
    let k = local("k", i64());
    let r = local("r", i64());
    let ok = local("ok", boolean());
    let tb = kept("t", table.clone());
    let mut d = Vec::new();

    // ─── the host's streams ─────────────────────────────────────
    for (fname, params, ret_ty, symbol) in [
        ("zl_io_error", vec![], string(), "$Lua$io_error"),
        ("zl_io_reason", vec![], string(), "$Lua$io_reason"),
        ("zl_io_errno", vec![], i64(), "$Lua$io_errno"),
        ("zl_io_failed", vec![], boolean(), "$Lua$io_failed"),
        ("zl_io_std", vec![("which", i64())], i64(), "$Lua$io_std"),
        (
            "zl_io_open",
            vec![("name", string()), ("mode", string())],
            i64(),
            "$Lua$io_open",
        ),
        (
            "zl_io_popen",
            vec![("command", string()), ("mode", string())],
            i64(),
            "$Lua$io_popen",
        ),
        ("zl_io_tmpfile", vec![], i64(), "$Lua$io_tmpfile"),
        (
            "zl_io_is_open",
            vec![("h", i64())],
            boolean(),
            "$Lua$io_is_open",
        ),
        (
            "zl_io_is_pipe",
            vec![("h", i64())],
            boolean(),
            "$Lua$io_is_pipe",
        ),
        ("zl_io_close", vec![("h", i64())], i64(), "$Lua$io_close"),
        (
            "zl_io_read_line",
            vec![("h", i64()), ("keep", boolean())],
            string(),
            "$Lua$io_read_line",
        ),
        (
            "zl_io_read_all",
            vec![("h", i64())],
            string(),
            "$Lua$io_read_all",
        ),
        (
            "zl_io_read_bytes",
            vec![("h", i64()), ("n", i64())],
            string(),
            "$Lua$io_read_bytes",
        ),
        (
            "zl_io_read_number",
            vec![("h", i64())],
            i64(),
            "$Lua$io_read_number",
        ),
        ("zl_io_number_int", vec![], i64(), "$Lua$io_number_int"),
        ("zl_io_number_float", vec![], f64(), "$Lua$io_number_float"),
        (
            "zl_io_write",
            vec![("h", i64()), ("s", string())],
            i64(),
            "$Lua$io_write",
        ),
        (
            "zl_io_seek",
            vec![("h", i64()), ("whence", i64()), ("offset", i64())],
            i64(),
            "$Lua$io_seek",
        ),
        ("zl_io_flush", vec![("h", i64())], i64(), "$Lua$io_flush"),
        ("zl_io_flush_all", vec![], unit(), "$Lua$io_flush_all"),
        (
            "zl_io_setvbuf",
            vec![("h", i64()), ("mode", i64()), ("size", i64())],
            i64(),
            "$Lua$io_setvbuf",
        ),
    ] {
        let params: Vec<(&str, Type)> = params.into_iter().collect();
        d.push(extern_fn(fname, &params, ret_ty, Some(symbol)));
    }

    // ─── files as values ────────────────────────────────────────
    let handle_of = |x: Expr| call("zb_unbox_instance_raw", vec![x], i64());
    d.push(define(
        "zl_file_new",
        &[&h],
        any(),
        vec![ret(call(
            "zb_box_instance",
            vec![h.e(), int32(file_tag() as i32)],
            any(),
        ))],
    ));
    let is_open = |x: Expr| call("zl_io_is_open", vec![handle_of(x)], boolean());
    d.push(define(
        "zl_file_is_open",
        &[&x],
        boolean(),
        vec![ret(and(is_file(x.e()), is_open(x.e())))],
    ));
    // The handle of a file that is open; `what` starts the message
    // for a value that is no file.
    d.push(define(
        "zl_file_check",
        &[&x, &what],
        i64(),
        vec![
            when(
                not(is_file(x.e())),
                vec![lua_error(concat(vec![
                    what.e(),
                    text(" (FILE* expected, got "),
                    arg_type_name(x.e()),
                    text(")"),
                ]))],
            ),
            when(
                not(is_open(x.e())),
                vec![lua_error(text("attempt to use a closed file"))],
            ),
            ret(handle_of(x.e())),
        ],
    ));
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
    // A file for a handle the host gave, or the failure it noted.
    d.push(define(
        "zl_file_or_failure",
        &[&h],
        any(),
        vec![
            when(eq(h.e(), int(0)), vec![failure()]),
            one(call("zl_file_new", vec![h.e()], any())),
        ],
    ));
    // `io.open(name, mode)`.
    d.push(define(
        "zl_io_open_of",
        &[&name, &mode],
        any(),
        vec![
            h.decl(call("zl_io_open", vec![name.e(), mode.e()], i64())),
            when(
                eq(h.e(), int(-1)),
                vec![lua_error(text("bad argument #2 to 'open' (invalid mode)"))],
            ),
            ret(call("zl_file_or_failure", vec![h.e()], any())),
        ],
    ));
    d.push(define(
        "zl_io_popen_of",
        &[&name, &mode],
        any(),
        vec![
            when(
                not(or(str_eq(mode.e(), text("r")), str_eq(mode.e(), text("w")))),
                vec![lua_error(text("bad argument #2 to 'popen' (invalid mode)"))],
            ),
            ret(call(
                "zl_file_or_failure",
                vec![call("zl_io_popen", vec![name.e(), mode.e()], i64())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_io_tmpfile_of",
        &[],
        any(),
        vec![ret(call(
            "zl_file_or_failure",
            vec![call("zl_io_tmpfile", vec![], i64())],
            any(),
        ))],
    ));
    // `io.type(x)`.
    d.push(define(
        "zl_io_type",
        &[&x],
        any(),
        vec![
            when(not(is_file(x.e())), vec![ret(nil())]),
            when(is_open(x.e()), vec![ret(box_str(text("file")))]),
            ret(box_str(text("closed file"))),
        ],
    ));
    // The standard streams as files, each made once.
    for (fname, which) in [("zl_io_stdin", 0), ("zl_io_stdout", 1), ("zl_io_stderr", 2)] {
        let cache = format!("{fname}_cache");
        d.push(global_var(&cache, any()));
        d.push(define(
            fname,
            &[],
            any(),
            vec![
                when(
                    is_nil(read_global(&cache, any())),
                    vec![set_global(
                        &cache,
                        call(
                            "zl_file_new",
                            vec![call("zl_io_std", vec![int(which)], i64())],
                            any(),
                        ),
                    )],
                ),
                ret(read_global(&cache, any())),
            ],
        ));
    }
    // The default input and output.
    d.push(global_var("zl_io_in", any()));
    d.push(global_var("zl_io_out", any()));
    for (fname, var, std) in [
        ("zl_io_input_file", "zl_io_in", "zl_io_stdin"),
        ("zl_io_output_file", "zl_io_out", "zl_io_stdout"),
    ] {
        d.push(define(
            fname,
            &[],
            any(),
            vec![
                when(
                    is_nil(read_global(var, any())),
                    vec![set_global(var, call(std, vec![], any()))],
                ),
                ret(read_global(var, any())),
            ],
        ));
    }
    // `io.input([f])` and `io.output([f])`: a name (a string or a
    // number) opens the file, an open file becomes the default,
    // anything else is refused as no file; the default is answered.
    for (fname, var, mode, what) in [
        ("zl_io_input", "zl_io_in", "r", "input"),
        ("zl_io_output", "zl_io_out", "w", "output"),
    ] {
        d.push(define(
            fname,
            &[&x],
            any(),
            vec![
                when(
                    and(not(is_nil(x.e())), not(is_file(x.e()))),
                    vec![
                        when(
                            not(or(
                                or(
                                    eq(category(x.e()), int(STR)),
                                    eq(category(x.e()), int(FLOAT)),
                                ),
                                or(
                                    eq(category(x.e()), int(INT)),
                                    eq(category(x.e()), int(UINT)),
                                ),
                            )),
                            vec![
                                expr(call("zl_file_check", vec![x.e(), bad_arg(1, what)], i64())),
                                ret(nil()),
                            ],
                        ),
                        name.decl(call("zl_arg_str", vec![x.e(), bad_arg(1, what)], string())),
                        h.decl(call("zl_io_open", vec![name.e(), text(mode)], i64())),
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
                        set_global(var, call("zl_file_new", vec![h.e()], any())),
                    ],
                ),
                when(
                    is_file(x.e()),
                    vec![
                        expr(call("zl_file_check", vec![x.e(), bad_arg(1, what)], i64())),
                        when(
                            call("zl_file_is_open", vec![x.e()], boolean()),
                            vec![set_global(var, x.e())],
                        ),
                    ],
                ),
                when(
                    is_nil(read_global(var, any())),
                    vec![set_global(
                        var,
                        call(
                            if what == "input" {
                                "zl_io_stdin"
                            } else {
                                "zl_io_stdout"
                            },
                            vec![],
                            any(),
                        ),
                    )],
                ),
                ret(read_global(var, any())),
            ],
        ));
    }
    // The default file for `io.read`, `io.write` and `io.lines`: an
    // error when it was closed.
    for (fname, getter, what) in [
        ("zl_io_default_in", "zl_io_input_file", "input"),
        ("zl_io_default_out", "zl_io_output_file", "output"),
    ] {
        d.push(define(
            fname,
            &[],
            any(),
            vec![
                x.decl(call(getter, vec![], any())),
                when(
                    not(is_open(x.e())),
                    vec![lua_error(text(&format!("default {what} file is closed")))],
                ),
                ret(x.e()),
            ],
        ));
    }

    // ─── reading and writing ────────────────────────────────────
    // `f:read(...)`: a value per format, the first that fails nil and
    // the last; the line without its end when no format is given.
    let read_line = |keep: bool| s.decl(call("zl_io_read_line", vec![h.e(), bool(keep)], string()));
    let string_or_nil = || if_expr(eq(s.e(), null(string())), nil(), box_str(s.e()));
    d.push(define(
        "zl_file_read",
        &[&x, &args],
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
        &[&x, &args],
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
        &[&x],
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
        &[&x, &y, &a],
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
        &[&x],
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
        &[&x, &y, &a],
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
        &[&x],
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
        &[&x],
        any(),
        vec![
            when(
                and(is_open(x.e()), gt(handle_of(x.e()), int(STDERR))),
                vec![expr(call("zl_io_close", vec![handle_of(x.e())], i64()))],
            ),
            ret(call("zl_none", vec![], any())),
        ],
    ));

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
        &[&env, &packed],
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
        &[&x, &args, &closing],
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
        &[&x, &args],
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
        &[&x, &args],
        any(),
        vec![ret(call(
            "zl_file_lines",
            vec![x.e(), args.e(), bool(false)],
            any(),
        ))],
    ));
    d.push(define(
        "zl_file_tostring",
        &[&x],
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
        &[&args],
        any(),
        vec![ret(call(
            "zl_file_read",
            vec![call("zl_io_default_in", vec![], any()), args.e()],
            any(),
        ))],
    ));
    d.push(define(
        "zl_io_write_of",
        &[&args],
        any(),
        vec![ret(call(
            "zl_file_write",
            vec![call("zl_io_default_out", vec![], any()), args.e()],
            any(),
        ))],
    ));
    d.push(define(
        "zl_io_close_of",
        &[&x],
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
        &[&x, &y],
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
    d
}
