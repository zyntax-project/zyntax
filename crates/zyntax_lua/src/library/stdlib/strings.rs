//! Strings declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        acc,
        anys,
        args,
        i,
        j,
        k,
        n,
        out,
        s,
        x,
        ..
    } = context;

    // ─── argument checks ────────────────────────────────────────
    // The message's start for the `i`th of a function's values, counted
    // from zero.
    let bad_arg_at = |i: Expr, name: &str| {
        concat(vec![
            text("bad argument #"),
            call("zb_str_of_int", vec![add(i, int(1))], string()),
            text(&format!(" to '{name}'")),
        ])
    };

    // ─── string ─────────────────────────────────────────────────
    for (name, params, ret_ty, symbol) in [
        (
            "zl_string_sub",
            vec![("s", string()), ("i", i64()), ("j", i64())],
            string(),
            "$Lua$sub",
        ),
        (
            "zl_byte_at",
            vec![("s", string()), ("i", i64())],
            i64(),
            "$Lua$byte_at",
        ),
        (
            "zl_from_byte",
            vec![("b", i64())],
            string(),
            "$Lua$from_byte",
        ),
        (
            "zl_string_reverse",
            vec![("s", string())],
            string(),
            "$Lua$reverse",
        ),
        (
            "zl_string_rep",
            vec![("s", string()), ("n", i64()), ("sep", string())],
            string(),
            "$Lua$rep",
        ),
        (
            "zl_find_plain",
            vec![("s", string()), ("p", string()), ("init", i64())],
            i64(),
            "$Lua$find_plain",
        ),
        ("zl_bytes", vec![("hex", string())], string(), "$Lua$bytes"),
        (
            "zl_string_upper",
            vec![("s", string())],
            string(),
            "$Lua$upper",
        ),
        (
            "zl_string_lower",
            vec![("s", string())],
            string(),
            "$Lua$lower",
        ),
        (
            "zl_format_raw",
            vec![
                ("fmt", string()),
                ("args", anys.clone()),
                ("texts", anys.clone()),
            ],
            string(),
            "$Lua$format",
        ),
        (
            "zl_pack_raw",
            vec![("fmt", string()), ("args", anys.clone())],
            string(),
            "$Lua$pack",
        ),
        ("zl_pack_error", vec![], string(), "$Lua$pack_error"),
        (
            "zl_packsize_raw",
            vec![("fmt", string())],
            i64(),
            "$Lua$packsize",
        ),
        (
            "zl_unpack_raw",
            vec![("fmt", string()), ("s", string()), ("pos", i64())],
            i64(),
            "$Lua$unpack",
        ),
        // The same over a string or a buffer (`Param::Bytes`).
        (
            "zl_string_len$any",
            vec![("s", any())],
            i64(),
            "$Lua$len_any",
        ),
        (
            "zl_string_sub$any",
            vec![("s", any()), ("i", i64()), ("j", i64())],
            string(),
            "$Lua$sub_any",
        ),
        (
            "zl_byte_at$any",
            vec![("s", any()), ("i", i64())],
            i64(),
            "$Lua$byte_at_any",
        ),
        (
            "zl_unpack_raw$any",
            vec![("fmt", string()), ("s", any()), ("pos", i64())],
            i64(),
            "$Lua$unpack_any",
        ),
        (
            "zl_unpack_kind",
            vec![("i", i64())],
            i64(),
            "$Lua$unpack_kind",
        ),
        (
            "zl_unpack_int",
            vec![("i", i64())],
            i64(),
            "$Lua$unpack_int",
        ),
        (
            "zl_unpack_float",
            vec![("i", i64())],
            f64(),
            "$Lua$unpack_float",
        ),
        (
            "zl_unpack_str",
            vec![("i", i64())],
            string(),
            "$Lua$unpack_str",
        ),
        ("zl_unpack_next", vec![], i64(), "$Lua$unpack_next"),
        ("zl_os_clock", vec![], f64(), "$Lua$clock"),
        ("zl_os_time", vec![], i64(), "$Lua$time"),
        ("zl_os_error", vec![], string(), "$Lua$os_error"),
        (
            "zl_date_field",
            vec![("t", i64()), ("utc", boolean()), ("index", i64())],
            i64(),
            "$Lua$date_field",
        ),
        (
            "zl_time_of",
            vec![
                ("year", i64()),
                ("month", i64()),
                ("day", i64()),
                ("hour", i64()),
                ("min", i64()),
                ("sec", i64()),
                ("isdst", i64()),
            ],
            i64(),
            "$Lua$time_of",
        ),
        (
            "zl_date_check",
            vec![("fmt", string())],
            string(),
            "$Lua$date_check",
        ),
        (
            "zl_date_raw",
            vec![("fmt", string()), ("t", i64()), ("utc", boolean())],
            string(),
            "$Lua$date",
        ),
        (
            "zl_getenv",
            vec![("name", string())],
            string(),
            "$Lua$getenv",
        ),
        ("zl_tmpname", vec![], string(), "$Lua$tmpname"),
        ("zl_remove", vec![("name", string())], i64(), "$Lua$remove"),
        (
            "zl_rename",
            vec![("from", string()), ("to", string())],
            i64(),
            "$Lua$rename",
        ),
        (
            "zl_execute",
            vec![("command", string())],
            i64(),
            "$Lua$execute",
        ),
        (
            "zl_exec_result",
            vec![("status", i64()), ("want_signal", boolean())],
            i64(),
            "$Lua$exec_result",
        ),
        (
            "zl_setlocale",
            vec![("locale", string())],
            string(),
            "$Lua$setlocale",
        ),
        (
            "zl_random_seed",
            vec![("n1", i64()), ("n2", i64())],
            unit(),
            "$Lua$random_seed",
        ),
        ("zl_random_next", vec![], i64(), "$Lua$random_next"),
        ("zl_random_seed_now", vec![], i64(), "$Lua$random_seed_now"),
        ("zl_random_seed_pid", vec![], i64(), "$Lua$random_seed_pid"),
        ("zl_random_float", vec![], f64(), "$Lua$random_float"),
        (
            "zl_random_int",
            vec![("lo", i64()), ("hi", i64())],
            i64(),
            "$Lua$random_int",
        ),
    ] {
        let params: Vec<(&str, Type)> = params.into_iter().collect();
        d.push(extern_fn(name, &params, ret_ty, Some(symbol)));
    }
    d.push(define(
        "zl_string_len",
        &[s],
        i64(),
        vec![ret(call("zb_str_len", vec![s.e()], i64()))],
    ));
    // `string.byte(s, i, j)`: the bytes at `i..=j`, `j` being `i` when
    // absent; its `$any` form reads a string or a buffer.
    let subject_any = kept("s", any());
    for (name, subject, len_of, byte_at) in [
        ("zl_string_byte", s, "zb_str_len", "zl_byte_at"),
        (
            "zl_string_byte$any",
            &subject_any,
            "zl_string_len$any",
            "zl_byte_at$any",
        ),
    ] {
        d.push(define(
            name,
            &[subject, i, j],
            any(),
            vec![
                when(eq(j.e(), int(i64::MIN)), vec![j.set(i.e())]),
                n.decl(call(len_of, vec![subject.e()], i64())),
                when(
                    lt(i.e(), int(0)),
                    vec![i.set(add(add(n.e(), i.e()), int(1)))],
                ),
                when(
                    lt(j.e(), int(0)),
                    vec![j.set(add(add(n.e(), j.e()), int(1)))],
                ),
                when(lt(i.e(), int(1)), vec![i.set(int(1))]),
                when(gt(j.e(), n.e()), vec![j.set(n.e())]),
                out.decl(list(vec![], anys.clone())),
                k.decl(i.e()),
                while_(
                    le(k.e(), j.e()),
                    vec![
                        push(
                            out.e(),
                            box_i64(call(byte_at, vec![subject.e(), k.e()], i64())),
                        ),
                        k.add_assign(int(1)),
                    ],
                ),
                ret(call("zb_box_tuple", vec![out.e()], any())),
            ],
        ));
    }
    d.push(define(
        "zl_string_char",
        &[args],
        string(),
        vec![
            acc.decl(text("")),
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    n.decl(call(
                        "zl_arg_int",
                        vec![at(args.e(), i.e()), bad_arg_at(i.e(), "char")],
                        i64(),
                    )),
                    when(
                        or(lt(n.e(), int(0)), gt(n.e(), int(255))),
                        vec![lua_error(add(
                            bad_arg_at(i.e(), "char"),
                            text(" (value out of range)"),
                        ))],
                    ),
                    acc.set(add(acc.e(), call("zl_from_byte", vec![n.e()], string()))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(acc.e()),
        ],
    ));
    // The host formats; a message it could not is prefixed by a byte
    // no text starts with.
    d.push(define(
        "zl_string_format",
        &[s, args],
        string(),
        vec![
            // An object's `%s` text is `tostring`'s, metamethods and
            // `__name` included, worked out here for the host.
            out.decl(list(vec![], anys.clone())),
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    x.decl(at(args.e(), i.e())),
                    if_(
                        and(not(is_nil(x.e())), eq(category(x.e()), int(CUSTOM))),
                        vec![push(
                            out.e(),
                            box_str(call("zl_tostring", vec![x.e()], string())),
                        )],
                        vec![push(out.e(), nil())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            acc.decl(call(
                "zl_format_raw",
                vec![s.e(), args.e(), out.e()],
                string(),
            )),
            when(
                eq(acc.e(), null(string())),
                vec![lua_error(call("zl_pack_error", vec![], string()))],
            ),
            ret(acc.e()),
        ],
    ));
}
