//! Files declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        h,
        mode,
        name,
        what,
        x,
        ..
    } = context;

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
        &[h],
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
        &[x],
        boolean(),
        vec![ret(and(is_file(x.e()), is_open(x.e())))],
    ));
    // The handle of a file that is open; `what` starts the message
    // for a value that is no file.
    d.push(define(
        "zl_file_check",
        &[x, what],
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
        &[h],
        any(),
        vec![
            when(eq(h.e(), int(0)), vec![failure()]),
            one(call("zl_file_new", vec![h.e()], any())),
        ],
    ));
    // `io.open(name, mode)`.
    d.push(define(
        "zl_io_open_of",
        &[name, mode],
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
        &[name, mode],
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
        &[x],
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
            &[x],
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
}
