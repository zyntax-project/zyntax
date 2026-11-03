//! Operating system declarations.

use super::*;

/// The `os` library above its host side: dates through the table
/// `os.date("*t")` gives and `os.time` takes, and the results a file
/// operation reports.
pub(super) fn os_declarations(t: &Types) -> Vec<Decl> {
    let table = t.table();
    let anys = t.anys();
    let x = kept("x", any());
    let y = kept("y", any());
    let tb = kept("t", table.clone());
    let key = kept("key", string());
    let default = local("default", i64());
    let delta = local("delta", i64());
    let v = kept("v", any());
    let n = local("n", i64());
    let time = local("time", i64());
    let utc = local("utc", boolean());
    let s = kept("s", string());
    let bad = kept("bad", string());
    let status = local("status", i64());
    let signalled = local("signalled", boolean());
    let how = kept("how", any());
    let mut d = Vec::new();
    // A field of a date table as an integer: `default` when absent,
    // required when that is negative; `delta` is what the C library
    // subtracts, checked to fit.
    d.push(define(
        "zl_date_table_field",
        &[&tb, &key, &default, &delta],
        i64(),
        vec![
            v.decl(call("zl_rawget_str", vec![tb.e(), key.e()], any())),
            when(
                is_nil(v.e()),
                vec![
                    when(
                        lt(default.e(), int(0)),
                        vec![lua_error(concat(vec![
                            text("field '"),
                            key.e(),
                            text("' missing in date table"),
                        ]))],
                    ),
                    ret(default.e()),
                ],
            ),
            y.decl(call("zl_math_tointeger", vec![v.e()], any())),
            when(
                is_nil(y.e()),
                vec![lua_error(concat(vec![
                    text("field '"),
                    key.e(),
                    text("' is not an integer"),
                ]))],
            ),
            n.decl(sub(get_i64(y.e()), delta.e())),
            when(
                or(
                    gt(n.e(), int(i32::MAX as i64)),
                    lt(n.e(), int(i32::MIN as i64)),
                ),
                vec![lua_error(concat(vec![
                    text("field '"),
                    key.e(),
                    text("' is out-of-bound"),
                ]))],
            ),
            ret(get_i64(y.e())),
        ],
    ));
    // The date table's fields set from a time, as `*t` lists them.
    let fields = ["year", "month", "day", "hour", "min", "sec", "wday", "yday"];
    let mut st = Vec::new();
    for (i, name) in fields.iter().enumerate() {
        st.push(expr(call(
            "zl_rawset_str",
            vec![
                tb.e(),
                text(name),
                box_i64(call(
                    "zl_date_field",
                    vec![time.e(), utc.e(), int(i as i64)],
                    i64(),
                )),
            ],
            unit(),
        )));
    }
    st.push(expr(call(
        "zl_rawset_str",
        vec![
            tb.e(),
            text("isdst"),
            box_bool(gt(
                call("zl_date_field", vec![time.e(), utc.e(), int(8)], i64()),
                int(0),
            )),
        ],
        unit(),
    )));
    st.push(ret_void());
    d.push(define(
        "zl_date_table_fill",
        &[&tb, &time, &utc],
        unit(),
        st,
    ));
    // `os.time([t])`: now, or the time a date table names, its fields
    // normalized in place.
    let field = |name: &str, default: i64, delta: i64| {
        call(
            "zl_date_table_field",
            vec![tb.e(), text(name), int(default), int(delta)],
            i64(),
        )
    };
    let isdst = local("isdst", i64());
    d.push(define(
        "zl_os_time_of",
        &[&x],
        i64(),
        vec![
            when(is_nil(x.e()), vec![ret(call("zl_os_time", vec![], i64()))]),
            when(
                not(is_table(x.e())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'time' (table expected, got "),
                    arg_type_name(x.e()),
                    text(")"),
                ]))],
            ),
            tb.decl(unbox_table(x.e(), t)),
            isdst.decl(int(-1)),
            v.decl(call("zl_rawget_str", vec![tb.e(), text("isdst")], any())),
            when(
                not(is_nil(v.e())),
                vec![isdst.set(if_expr(
                    call("zl_truthy", vec![v.e()], boolean()),
                    int(1),
                    int(0),
                ))],
            ),
            time.decl(call(
                "zl_time_of",
                vec![
                    field("year", -1, 1900),
                    field("month", -1, 1),
                    field("day", -1, 0),
                    field("hour", 12, 0),
                    field("min", 0, 0),
                    field("sec", 0, 0),
                    isdst.e(),
                ],
                i64(),
            )),
            when(
                eq(time.e(), int(-1)),
                vec![lua_error(text(
                    "time result cannot be represented in this installation",
                ))],
            ),
            expr(call(
                "zl_date_table_fill",
                vec![tb.e(), time.e(), bool(false)],
                unit(),
            )),
            ret(time.e()),
        ],
    ));
    // `os.date([format [, time]])`: `!` for UTC, `*t` for a table,
    // otherwise `strftime` with each conversion checked first.
    d.push(define(
        "zl_os_date",
        &[&x, &y],
        any(),
        vec![
            s.decl(if_expr(
                is_nil(x.e()),
                text("%c"),
                call("zl_arg_str", vec![x.e(), bad_arg(1, "date")], string()),
            )),
            time.decl(if_expr(
                is_nil(y.e()),
                call("zl_os_time", vec![], i64()),
                call("zl_arg_int", vec![y.e(), bad_arg(2, "date")], i64()),
            )),
            utc.decl(call("zb_str_startswith", vec![s.e(), text("!")], boolean())),
            when(
                utc.e(),
                vec![s.set(call(
                    "zb_str_substring",
                    vec![s.e(), int(1), call("zb_str_len", vec![s.e()], i64())],
                    string(),
                ))],
            ),
            when(
                eq(
                    call("zl_date_field", vec![time.e(), utc.e(), int(0)], i64()),
                    int(i64::MIN),
                ),
                vec![lua_error(text(
                    "date result cannot be represented in this installation",
                ))],
            ),
            when(
                call("zb_str_startswith", vec![s.e(), text("*t")], boolean()),
                vec![
                    tb.decl(call("zl_table_new", vec![], table.clone())),
                    expr(call(
                        "zl_date_table_fill",
                        vec![tb.e(), time.e(), utc.e()],
                        unit(),
                    )),
                    ret(box_table(tb.e())),
                ],
            ),
            bad.decl(call("zl_date_check", vec![s.e()], string())),
            when(
                ne(bad.e(), null(string())),
                vec![lua_error(concat(vec![
                    text("bad argument #1 to 'date' (invalid conversion specifier '%"),
                    bad.e(),
                    text("')"),
                ]))],
            ),
            ret(box_str(call(
                "zl_date_raw",
                vec![s.e(), time.e(), utc.e()],
                string(),
            ))),
        ],
    ));
    let a = local("a", i64());
    let b = local("b", i64());
    d.push(define(
        "zl_os_difftime",
        &[&a, &b],
        f64(),
        vec![ret(sub(cast(a.e(), f64()), cast(b.e(), f64())))],
    ));
    d.push(define(
        "zl_os_getenv",
        &[&s],
        any(),
        vec![
            key.decl(call("zl_getenv", vec![s.e()], string())),
            when(eq(key.e(), null(string())), vec![ret(nil())]),
            ret(box_str(key.e())),
        ],
    ));
    d.push(define(
        "zl_os_tmpname",
        &[],
        string(),
        vec![
            key.decl(call("zl_tmpname", vec![], string())),
            when(
                eq(key.e(), null(string())),
                vec![lua_error(text("unable to generate a unique filename"))],
            ),
            ret(key.e()),
        ],
    ));
    // What a file operation answers: true, or nil, the message and
    // the error's number.
    let file_result = |result: Expr| {
        vec![
            status.decl(result),
            when(
                eq(status.e(), int(0)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![box_bool(bool(true))], anys.clone())],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        nil(),
                        box_str(call("zl_os_error", vec![], string())),
                        box_i64(status.e()),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ]
    };
    d.push(define(
        "zl_os_remove",
        &[&s],
        any(),
        file_result(call("zl_remove", vec![s.e()], i64())),
    ));
    d.push(define(
        "zl_os_rename",
        &[&s, &key],
        any(),
        file_result(call("zl_rename", vec![s.e(), key.e()], i64())),
    ));
    // `os.execute([command])`: whether a shell is there, or how the
    // command ended: true or nil, "exit" or "signal", and the number.
    d.push(define(
        "zl_os_execute",
        &[&x],
        any(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(box_bool(ne(
                    call("zl_execute", vec![null(string())], i64()),
                    int(0),
                )))],
            ),
            status.decl(call(
                "zl_execute",
                vec![call(
                    "zl_arg_str",
                    vec![x.e(), bad_arg(1, "execute")],
                    string(),
                )],
                i64(),
            )),
            when(
                eq(status.e(), int(-1)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_str(text("exit")), box_i64(int(-1))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            n.decl(call("zl_exec_result", vec![status.e(), bool(false)], i64())),
            signalled.decl(ne(
                call("zl_exec_result", vec![status.e(), bool(true)], i64()),
                int(0),
            )),
            v.decl(nil()),
            when(
                and(not(signalled.e()), eq(n.e(), int(0))),
                vec![v.set(box_bool(bool(true)))],
            ),
            how.decl(box_str(if_expr(
                signalled.e(),
                text("signal"),
                text("exit"),
            ))),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![v.e(), how.e(), box_i64(n.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_os_setlocale",
        &[&x, &y],
        any(),
        vec![
            key.decl(call(
                "zl_setlocale",
                vec![if_expr(
                    is_nil(x.e()),
                    null(string()),
                    call("zl_arg_str", vec![x.e(), bad_arg(1, "setlocale")], string()),
                )],
                string(),
            )),
            when(eq(key.e(), null(string())), vec![ret(nil())]),
            ret(box_str(key.e())),
        ],
    ));
    d
}
