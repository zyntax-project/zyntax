//! Comparison declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        a,
        b,
        ca,
        cb,
        f,
        fa,
        h,
        ia,
        ..
    } = context;

    // ─── comparison ─────────────────────────────────────────────
    // An integer against a float, exactly: the float is brought to an
    // integer when it has one in range, otherwise its side of the
    // range decides. An integer within 2^53 compares as a float.
    let fits_float = |i: Expr| and(ge(i.clone(), int(-(1i64 << 53))), le(i, int(1i64 << 53)));
    let in_range = |x: Expr| {
        and(
            ge(x.clone(), float(-9223372036854775808.0)),
            lt(x, float(9223372036854775808.0)),
        )
    };
    d.push(define(
        "zl_eq_if",
        &[ia, fa],
        boolean(),
        vec![
            // An integer a float holds exactly equals the float that is
            // it, and nothing else.
            when(
                fits_float(ia.e()),
                vec![ret(eq(cast(ia.e(), f64()), fa.e()))],
            ),
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                ne(call("floor", vec![fa.e()], f64()), fa.e()),
                vec![ret(bool(false))],
            ),
            when(not(in_range(fa.e())), vec![ret(bool(false))]),
            ret(eq(ia.e(), cast(fa.e(), i64()))),
        ],
    ));
    // `i < f`: with the float's ceiling; `i <= f`: with its floor.
    d.push(define(
        "zl_lt_if",
        &[ia, fa],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(lt(cast(ia.e(), f64()), fa.e()))],
            ),
            f.decl(call("zl_ceil_f64", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(lt(ia.e(), cast(f.e(), i64())))]),
            ret(gt(fa.e(), float(0.0))),
        ],
    ));
    d.push(define(
        "zl_le_if",
        &[ia, fa],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(le(cast(ia.e(), f64()), fa.e()))],
            ),
            f.decl(call("floor", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(le(ia.e(), cast(f.e(), i64())))]),
            ret(gt(fa.e(), float(0.0))),
        ],
    ));
    // `f < i`: with the float's floor; `f <= i`: with its ceiling.
    d.push(define(
        "zl_lt_fi",
        &[fa, ia],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(lt(fa.e(), cast(ia.e(), f64())))],
            ),
            f.decl(call("floor", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(lt(cast(f.e(), i64()), ia.e()))]),
            ret(lt(fa.e(), float(0.0))),
        ],
    ));
    d.push(define(
        "zl_le_fi",
        &[fa, ia],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(le(fa.e(), cast(ia.e(), f64())))],
            ),
            f.decl(call("zl_ceil_f64", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(le(cast(f.e(), i64()), ia.e()))]),
            ret(lt(fa.e(), float(0.0))),
        ],
    ));
    let num_eq = |a: Expr, b: Expr| {
        if_expr(
            and(is_int_cat_of(a.clone()), is_int_cat_of(b.clone())),
            eq(get_i64(a.clone()), get_i64(b.clone())),
            if_expr(
                is_int_cat_of(a.clone()),
                call(
                    "zl_eq_if",
                    vec![get_i64(a.clone()), get_f64(b.clone())],
                    boolean(),
                ),
                if_expr(
                    is_int_cat_of(b.clone()),
                    call(
                        "zl_eq_if",
                        vec![get_i64(b.clone()), get_f64(a.clone())],
                        boolean(),
                    ),
                    eq(get_f64(a), get_f64(b)),
                ),
            ),
        )
    };
    d.push(define(
        "zl_eq",
        &[a, b],
        boolean(),
        vec![
            when(and(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(true))]),
            when(or(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(false))]),
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            // Numbers compare by value first: a NaN is not itself,
            // whichever box holds it.
            when(
                and(is_number_cat(ca), is_number_cat(cb)),
                vec![ret(num_eq(a.e(), b.e()))],
            ),
            when(eq(a.e(), b.e()), vec![ret(bool(true))]),
            when(
                or(
                    zyntax_builtins::foreign::is_foreign(a.e()),
                    zyntax_builtins::foreign::is_foreign(b.e()),
                ),
                vec![ret(and(
                    and(
                        zyntax_builtins::foreign::is_foreign(a.e()),
                        zyntax_builtins::foreign::is_foreign(b.e()),
                    ),
                    call("zb_foreign_eq", vec![a.e(), b.e()], boolean()),
                ))],
            ),
            when(
                and(is_cat(ca, STR), is_cat(cb, STR)),
                vec![ret(str_eq(get_str(a.e()), get_str(b.e())))],
            ),
            when(
                and(is_cat(ca, BOOL), is_cat(cb, BOOL)),
                vec![ret(eq(get_bool(a.e()), get_bool(b.e())))],
            ),
            when(
                or(ne(ca.e(), int(CUSTOM)), ne(cb.e(), int(CUSTOM))),
                vec![ret(bool(false))],
            ),
            when(
                eq(
                    call("zb_unbox_instance_raw", vec![a.e()], i64()),
                    call("zb_unbox_instance_raw", vec![b.e()], i64()),
                ),
                vec![ret(bool(true))],
            ),
            when(
                and(is_table(a.e()), is_table(b.e())),
                vec![
                    h.decl(call("zl_meta_of", vec![a.e(), text("__eq")], any())),
                    when(
                        is_nil(h.e()),
                        vec![h.set(call("zl_meta_of", vec![b.e(), text("__eq")], any()))],
                    ),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            metamethod_call_check(h.e(), text("eq")),
                            ret(call(
                                "zl_truthy",
                                vec![call(
                                    "zl_first",
                                    vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                                    any(),
                                )],
                                boolean(),
                            )),
                        ],
                    ),
                ],
            ),
            ret(bool(false)),
        ],
    ));
    let compare_error = |a: &Local, b: &Local| {
        vec![
            if_(
                str_eq(obj_type_name(a.e()), obj_type_name(b.e())),
                vec![lua_error(concat(vec![
                    text("attempt to compare two "),
                    obj_type_name(a.e()),
                    text(" values"),
                ]))],
                vec![lua_error(concat(vec![
                    text("attempt to compare "),
                    obj_type_name(a.e()),
                    text(" with "),
                    obj_type_name(b.e()),
                ]))],
            ),
            ret(bool(false)),
        ]
    };
    let order = |name: &str,
                 event: &str,
                 int_cmp: fn(Expr, Expr) -> Expr,
                 str_test: fn(Expr) -> Expr,
                 mixed: (&str, &str)| {
        let mut st = vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                and(is_number_cat(ca), is_number_cat(cb)),
                vec![
                    when(
                        and(is_int_cat_of(a.e()), is_int_cat_of(b.e())),
                        vec![ret(int_cmp(get_i64(a.e()), get_i64(b.e())))],
                    ),
                    when(
                        is_int_cat_of(a.e()),
                        vec![ret(call(
                            mixed.0,
                            vec![get_i64(a.e()), get_f64(b.e())],
                            boolean(),
                        ))],
                    ),
                    when(
                        is_int_cat_of(b.e()),
                        vec![ret(call(
                            mixed.1,
                            vec![get_f64(a.e()), get_i64(b.e())],
                            boolean(),
                        ))],
                    ),
                    ret(int_cmp(get_f64(a.e()), get_f64(b.e()))),
                ],
            ),
            when(
                and(is_cat(ca, STR), is_cat(cb, STR)),
                vec![ret(str_test(cast(
                    call("zb_str_cmp", vec![get_str(a.e()), get_str(b.e())], i32()),
                    i64(),
                )))],
            ),
            h.decl(call("zl_meta_of", vec![a.e(), text(event)], any())),
            when(
                is_nil(h.e()),
                vec![h.set(call("zl_meta_of", vec![b.e(), text(event)], any()))],
            ),
            when(
                not(is_nil(h.e())),
                vec![
                    metamethod_call_check(h.e(), text(&event[2..])),
                    ret(call(
                        "zl_truthy",
                        vec![call(
                            "zl_first",
                            vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                            any(),
                        )],
                        boolean(),
                    )),
                ],
            ),
        ];
        st.extend(compare_error(a, b));
        define(name, &[a, b], boolean(), st)
    };
    d.push(order(
        "zl_lt",
        "__lt",
        lt,
        |c| lt(c, int(0)),
        ("zl_lt_if", "zl_lt_fi"),
    ));
    d.push(order(
        "zl_le",
        "__le",
        le,
        |c| le(c, int(0)),
        ("zl_le_if", "zl_le_fi"),
    ));
}
