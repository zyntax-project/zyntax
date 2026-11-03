//! Conversions declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        cat,
        h,
        i,
        n,
        nb,
        r,
        s,
        x,
        ..
    } = context;
    let _ = &nb;

    // ─── the host's number primitives ───────────────────────────
    d.push(extern_fn(
        "zl_float_str",
        &[("x", f64())],
        string(),
        Some("$Lua$float_str"),
    ));
    d.push(extern_fn(
        "zl_number_kind",
        &[("s", string())],
        i64(),
        Some("$Lua$number_kind"),
    ));
    d.push(extern_fn(
        "zl_parse_int",
        &[("s", string())],
        i64(),
        Some("$Lua$parse_int"),
    ));
    d.push(extern_fn(
        "zl_parse_float",
        &[("s", string())],
        f64(),
        Some("$Lua$parse_float"),
    ));
    d.push(extern_fn(
        "zl_float_is_int",
        &[("x", f64())],
        boolean(),
        Some("$Lua$float_is_int"),
    ));

    // ─── truth ──────────────────────────────────────────────────
    // Only nil and false are false.
    d.push(define(
        "zl_truthy",
        &[x],
        boolean(),
        vec![
            when(is_nil(x.e()), vec![ret(bool(false))]),
            when(eq(category(x.e()), int(BOOL)), vec![ret(get_bool(x.e()))]),
            ret(bool(true)),
        ],
    ));

    // ─── conversions ────────────────────────────────────────────
    // A number's text: integers in decimal, floats as `%.14g`.
    d.push(define(
        "zl_number_str",
        &[x],
        string(),
        vec![
            cat.decl(category(x.e())),
            when(
                is_int_cat(cat),
                vec![ret(call("zb_str_of_int", vec![get_i64(x.e())], string()))],
            ),
            ret(call("zl_float_str", vec![get_f64(x.e())], string())),
        ],
    ));
    // The text of a function value: its code address.
    let func_str = |x: Expr| {
        add(
            text("function: 0x"),
            call(
                "zb_str_of_int_radix",
                vec![call("zb_unbox_instance_raw", vec![x], i64()), int32(16)],
                string(),
            ),
        )
    };
    let is_func = |x: Expr| eq(tag_of(x), int(zyntax_builtins::FUNC_TAG));
    let is_code = |x: Expr| eq(tag_of(x), int(zyntax_builtins::CODE_TAG));
    // A table's or a full userdata's text: its `__tostring`, else its
    // `__name` and address, else its type and address.
    let described = |x: &Local| -> Vec<Stmt> {
        vec![
            h.decl(call("zl_meta_of", vec![x.e(), text("__tostring")], any())),
            when(
                not(is_nil(h.e())),
                vec![
                    r.decl(call(
                        "zl_first",
                        vec![call("zl_call_1", vec![h.e(), x.e()], any())],
                        any(),
                    )),
                    when(
                        and(
                            not(is_nil(r.e())),
                            or(
                                eq(category(r.e()), int(INT)),
                                or(
                                    eq(category(r.e()), int(UINT)),
                                    eq(category(r.e()), int(FLOAT)),
                                ),
                            ),
                        ),
                        vec![ret(call("zl_number_str", vec![r.e()], string()))],
                    ),
                    when(
                        or(is_nil(r.e()), ne(category(r.e()), int(STR))),
                        vec![lua_error(text("'__tostring' must return a string"))],
                    ),
                    ret(get_str(r.e())),
                ],
            ),
            h.set(call("zl_meta_of", vec![x.e(), text("__name")], any())),
            when(
                eq(category(h.e()), int(STR)),
                vec![ret(add(
                    add(get_str(h.e()), text(": 0x")),
                    call(
                        "zb_str_of_int_radix",
                        vec![call("zb_unbox_instance_raw", vec![x.e()], i64()), int32(16)],
                        string(),
                    ),
                ))],
            ),
            ret(call("zb_hook_instance_str", vec![x.e()], string())),
        ]
    };
    d.push(define(
        "zl_tostring",
        &[x],
        string(),
        vec![
            // A value of a type `debug.setmetatable` gave a metatable:
            // its `__tostring`.
            when(
                and(
                    read_global(debug::TYPE_METAS_ON, boolean()),
                    and(
                        not(is_table(x.e())),
                        or(is_nil(x.e()), ne(category(x.e()), int(STR))),
                    ),
                ),
                vec![
                    h.decl(call("zl_meta_of", vec![x.e(), text("__tostring")], any())),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            r.decl(call(
                                "zl_first",
                                vec![call("zl_call_1", vec![h.e(), x.e()], any())],
                                any(),
                            )),
                            when(
                                or(is_nil(r.e()), ne(category(r.e()), int(STR))),
                                vec![lua_error(text("'__tostring' must return a string"))],
                            ),
                            ret(get_str(r.e())),
                        ],
                    ),
                ],
            ),
            when(is_nil(x.e()), vec![ret(text("nil"))]),
            when(
                zyntax_builtins::foreign::is_foreign(x.e()),
                vec![ret(call("zb_foreign_str", vec![x.e()], string()))],
            ),
            cat.decl(category(x.e())),
            when(is_cat(cat, STR), vec![ret(get_str(x.e()))]),
            when(
                is_number_cat(cat),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            when(
                is_cat(cat, BOOL),
                vec![ret(if_expr(get_bool(x.e()), text("true"), text("false")))],
            ),
            when(or(is_table(x.e()), is_userdata(x.e())), described(x)),
            when(
                or(is_thread(x.e()), is_file(x.e())),
                vec![ret(call("zb_hook_instance_str", vec![x.e()], string()))],
            ),
            when(
                or(is_func(x.e()), is_code(x.e())),
                vec![ret(func_str(x.e()))],
            ),
            ret(call("zb_any_str", vec![x.e()], string())),
        ],
    ));
    // `tonumber(x)`: a number as it is, a numeral's value, else nil.
    d.push(define(
        "zl_str_to_number",
        &[s],
        any(),
        vec![
            n.decl(call("zl_number_kind", vec![s.e()], i64())),
            when(
                eq(n.e(), int(1)),
                vec![ret(box_i64(call("zl_parse_int", vec![s.e()], i64())))],
            ),
            when(
                eq(n.e(), int(2)),
                vec![ret(box_f64(call("zl_parse_float", vec![s.e()], f64())))],
            ),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_tonumber",
        &[x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            cat.decl(category(x.e())),
            when(is_number_cat(cat), vec![ret(x.e())]),
            when(
                is_cat(cat, STR),
                vec![ret(call("zl_str_to_number", vec![get_str(x.e())], any()))],
            ),
            ret(nil()),
        ],
    ));
    // `tonumber(s, base)` for an integer numeral in another base.
    let base = local("base", i64());
    let digit = local("digit", i64());
    let acc = local("acc", i64());
    let neg = local("neg", boolean());
    let code = local("code", i64());
    let text_in = kept("text", string());
    d.push(define(
        "zl_tonumber_base",
        &[&text_in, &base],
        any(),
        vec![
            when(
                or(lt(base.e(), int(2)), gt(base.e(), int(36))),
                vec![lua_error(text(
                    "bad argument #2 to 'tonumber' (base out of range)",
                ))],
            ),
            s.decl(call(
                "zb_str_lower",
                vec![call("zb_str_strip", vec![text_in.e()], string())],
                string(),
            )),
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            when(eq(n.e(), int(0)), vec![ret(nil())]),
            i.decl(int(0)),
            neg.decl(bool(false)),
            code.decl(cast(
                call("zb_str_code_at", vec![s.e(), int(0)], i32()),
                i64(),
            )),
            when(
                eq(code.e(), int(45)),
                vec![neg.set(bool(true)), i.set(int(1))],
            ),
            when(eq(code.e(), int(43)), vec![i.set(int(1))]),
            when(ge(i.e(), n.e()), vec![ret(nil())]),
            acc.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    code.set(cast(
                        call("zb_str_code_at", vec![s.e(), i.e()], i32()),
                        i64(),
                    )),
                    digit.decl(int(-1)),
                    when(
                        and(ge(code.e(), int(48)), le(code.e(), int(57))),
                        vec![digit.set(sub(code.e(), int(48)))],
                    ),
                    when(
                        and(ge(code.e(), int(97)), le(code.e(), int(122))),
                        vec![digit.set(add(sub(code.e(), int(97)), int(10)))],
                    ),
                    when(
                        or(lt(digit.e(), int(0)), ge(digit.e(), base.e())),
                        vec![ret(nil())],
                    ),
                    acc.set(add(mul(acc.e(), base.e()), digit.e())),
                    i.add_assign(int(1)),
                ],
            ),
            when(neg.e(), vec![acc.set(sub(int(0), acc.e()))]),
            ret(box_i64(acc.e())),
        ],
    ));
}
