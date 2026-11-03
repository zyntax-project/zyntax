//! Checks declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        f,
        table,
        what,
        x,
        y,
        ..
    } = context;
    d.push(extern_fn(
        "zl_stdout_write",
        &[("s", string())],
        i64(),
        Some("$Lua$stdout_write"),
    ));
    // `what` starts the message: `bad argument #n to 'f'`.
    let arg_error = |what: &Local, expected: &str, got: Expr| {
        lua_error(concat(vec![
            what.e(),
            text(&format!(" ({expected} expected, got ")),
            got,
            text(")"),
        ]))
    };
    let got_of = |x: &Local| arg_type_name(x.e());
    d.push(define(
        "zl_arg_int",
        &[x, what],
        i64(),
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(is_nil(y.e()), vec![arg_error(what, "number", got_of(x))]),
            when(
                or(
                    eq(category(y.e()), int(INT)),
                    eq(category(y.e()), int(UINT)),
                ),
                vec![ret(get_i64(y.e()))],
            ),
            f.decl(get_f64(y.e())),
            when(
                not(call("zl_float_is_int", vec![f.e()], boolean())),
                vec![lua_error(add(
                    what.e(),
                    text(" (number has no integer representation)"),
                ))],
            ),
            ret(cast(f.e(), i64())),
        ],
    ));
    d.push(define(
        "zl_arg_float",
        &[x, what],
        f64(),
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(is_nil(y.e()), vec![arg_error(what, "number", got_of(x))]),
            when(
                or(
                    eq(category(y.e()), int(INT)),
                    eq(category(y.e()), int(UINT)),
                ),
                vec![ret(cast(get_i64(y.e()), f64()))],
            ),
            ret(get_f64(y.e())),
        ],
    ));
    d.push(define(
        "zl_arg_str",
        &[x, what],
        string(),
        vec![
            when(is_nil(x.e()), vec![arg_error(what, "string", got_of(x))]),
            when(eq(category(x.e()), int(STR)), vec![ret(get_str(x.e()))]),
            when(
                or(
                    or(
                        eq(category(x.e()), int(INT)),
                        eq(category(x.e()), int(UINT)),
                    ),
                    eq(category(x.e()), int(FLOAT)),
                ),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            arg_error(what, "string", got_of(x)),
            ret(text("")),
        ],
    ));
    // The subject of a function that reads bytes: a string as it is, a
    // number as its text, a foreign object that is a buffer as itself.
    d.push(define(
        "zl_arg_bytes",
        &[x, what],
        any(),
        vec![
            when(is_nil(x.e()), vec![arg_error(what, "string", got_of(x))]),
            when(eq(category(x.e()), int(STR)), vec![ret(x.e())]),
            when(
                or(
                    or(
                        eq(category(x.e()), int(INT)),
                        eq(category(x.e()), int(UINT)),
                    ),
                    eq(category(x.e()), int(FLOAT)),
                ),
                vec![ret(box_str(call("zl_number_str", vec![x.e()], string())))],
            ),
            when(
                and(
                    zyntax_builtins::foreign::is_foreign(x.e()),
                    ge(call("zb_foreign_bytes_len", vec![x.e()], i64()), int(0)),
                ),
                vec![ret(x.e())],
            ),
            arg_error(what, "string", got_of(x)),
            ret(nil()),
        ],
    ));
    // An optional integer: the default when absent.
    let default = local("default", i64());
    d.push(define(
        "zl_arg_opt_int",
        &[x, &default, what],
        i64(),
        vec![
            when(is_nil(x.e()), vec![ret(default.e())]),
            ret(call("zl_arg_int", vec![x.e(), what.e()], i64())),
        ],
    ));
    let default_s = kept("default", string());
    d.push(define(
        "zl_arg_opt_str",
        &[x, &default_s, what],
        string(),
        vec![
            when(is_nil(x.e()), vec![ret(default_s.e())]),
            ret(call("zl_arg_str", vec![x.e(), what.e()], string())),
        ],
    ));
    // An argument that was not passed at all, which the caller can
    // tell from nil: the error, then a stand-in.
    let missing = |what: &Local, expected: &str| {
        lua_error(add(
            what.e(),
            text(&format!(" ({expected} expected, got no value)")),
        ))
    };
    d.push(define_cold(
        "zl_arg_value_missing",
        &[what],
        any(),
        vec![
            lua_error(add(what.e(), text(" (value expected)"))),
            ret(nil()),
        ],
    ));
    let kind = kept("kind", string());
    d.push(define_cold(
        "zl_arg_expected_missing",
        &[what, &kind],
        any(),
        vec![
            lua_error(concat(vec![
                what.e(),
                text(" ("),
                kind.e(),
                text(" expected, got no value)"),
            ])),
            ret(nil()),
        ],
    ));
    d.push(define_cold(
        "zl_arg_int_missing",
        &[what],
        i64(),
        vec![missing(what, "number"), ret(int(0))],
    ));
    d.push(define_cold(
        "zl_arg_float_missing",
        &[what],
        f64(),
        vec![missing(what, "number"), ret(float(0.0))],
    ));
    d.push(define_cold(
        "zl_arg_str_missing",
        &[what],
        string(),
        vec![missing(what, "string"), ret(text(""))],
    ));
    d.push(define_cold(
        "zl_as_table_missing",
        &[what],
        table.clone(),
        vec![
            missing(what, "table"),
            ret(call("zl_table_new", vec![], table.clone())),
        ],
    ));
}
