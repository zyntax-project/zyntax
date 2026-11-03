//! Strings declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        a,
        b,
        cat,
        h,
        i,
        r,
        x,
        ..
    } = context;
    let is_text_like = |x: Expr| {
        let c = category(x);
        or(
            eq(c.clone(), int(STR)),
            or(
                eq(c.clone(), int(INT)),
                or(eq(c.clone(), int(UINT)), eq(c, int(FLOAT))),
            ),
        )
    };

    // ─── concatenation and length ───────────────────────────────
    // The text an operand of `..` contributes: strings as they are,
    // numbers as their text; anything else is the metamethod's.
    d.push(define(
        "zl_concat_text",
        &[x],
        string(),
        vec![
            when(
                is_nil(x.e()),
                vec![
                    type_error(
                        text("attempt to concatenate a nil value"),
                        int(OPERAND_LEFT),
                    ),
                    ret(text("")),
                ],
            ),
            cat.decl(category(x.e())),
            when(is_cat(cat, STR), vec![ret(get_str(x.e()))]),
            when(
                is_number_cat(cat),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            type_error(
                concat(vec![
                    text("attempt to concatenate a "),
                    obj_type_name(x.e()),
                    text(" value"),
                ]),
                int(OPERAND_LEFT),
            ),
            ret(text("")),
        ],
    ));
    d.push(define(
        "zl_concat",
        &[a, b],
        any(),
        vec![
            when(
                and(
                    and(not(is_nil(a.e())), not(is_nil(b.e()))),
                    and(is_text_like(a.e()), is_text_like(b.e())),
                ),
                vec![ret(call(
                    "zb_str_concat_boxed",
                    vec![
                        call("zl_concat_text", vec![a.e()], string()),
                        call("zl_concat_text", vec![b.e()], string()),
                    ],
                    any(),
                ))],
            ),
            ret(call(
                "zl_arith_meta",
                vec![int(OP_CONCAT), a.e(), b.e()],
                any(),
            )),
        ],
    ));
    // `#x`: a string's bytes, a table's border or its `__len`, a
    // buffer's bytes.
    d.push(define(
        "zl_len_any",
        &[x],
        any(),
        vec![
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(STR))),
                vec![ret(box_i64(call(
                    "zb_str_len",
                    vec![get_str(x.e())],
                    i64(),
                )))],
            ),
            when(
                and(
                    zyntax_builtins::foreign::is_foreign(x.e()),
                    ge(call("zb_foreign_bytes_len", vec![x.e()], i64()), int(0)),
                ),
                vec![ret(box_i64(call(
                    "zb_foreign_bytes_len",
                    vec![x.e()],
                    i64(),
                )))],
            ),
            when(
                is_table(x.e()),
                vec![
                    h.decl(call("zl_meta_of", vec![x.e(), text("__len")], any())),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            metamethod_call_check(h.e(), text("len")),
                            // The operand is passed twice, as for
                            // every unary event.
                            ret(call(
                                "zl_first",
                                vec![call("zl_call_2", vec![h.e(), x.e(), x.e()], any())],
                                any(),
                            )),
                        ],
                    ),
                    ret(box_i64(call("zl_len", vec![unbox_table(x.e(), t)], i64()))),
                ],
            ),
            // Another type's `__len`, from `debug.setmetatable`.
            h.decl(call("zl_meta_of", vec![x.e(), text("__len")], any())),
            when(
                not(is_nil(h.e())),
                vec![
                    metamethod_call_check(h.e(), text("len")),
                    ret(call(
                        "zl_first",
                        vec![call("zl_call_2", vec![h.e(), x.e(), x.e()], any())],
                        any(),
                    )),
                ],
            ),
            type_error(
                concat(vec![
                    text("attempt to get length of a "),
                    obj_type_name(x.e()),
                    text(" value"),
                ]),
                int(OPERAND_LEFT),
            ),
            ret(nil()),
        ],
    ));
    let _ = (&i, &r);
}
