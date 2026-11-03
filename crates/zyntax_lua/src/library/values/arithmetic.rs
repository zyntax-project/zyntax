//! Arithmetic declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        a,
        b,
        cat,
        f,
        fa,
        fb,
        h,
        ia,
        ib,
        na,
        nb,
        op,
        s,
        x,
        ..
    } = context;

    // ─── arithmetic ─────────────────────────────────────────────
    // The number an arithmetic operand is: itself, or the numeral a
    // string holds; nil when it is neither.
    d.push(define(
        "zl_arith_operand",
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
    // The event's name by code, or the operation's without the
    // underscores; a chain of comparisons keeps it in typed AST.
    let name_chain = |op: Expr, skip: usize| {
        let mut e = text(&EVENTS[EVENTS.len() - 1][skip..]);
        for (i, name) in EVENTS.iter().enumerate().rev().skip(1) {
            e = if_expr(eq(op.clone(), int(i as i64)), text(&name[skip..]), e);
        }
        e
    };
    let event_name = |op: Expr| name_chain(op, 0);
    let operation_name = |op: Expr| name_chain(op, 2);
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
    // Whether a number has an integer value.
    let has_int = |x: Expr| {
        or(
            is_int_cat_of(x.clone()),
            call("zl_float_is_int", vec![get_f64(x)], boolean()),
        )
    };
    let is_str = |x: Expr| and(not(is_nil(x.clone())), eq(category(x), int(STR)));
    let is_number = |x: Expr| {
        and(
            not(is_nil(x.clone())),
            or(is_int_cat_of(x.clone()), eq(category(x), int(FLOAT))),
        )
    };
    // The metamethod for `op` on `a` or `b`, called; or the error,
    // about the first operand that is wrong: for `..` the first that is
    // not text, otherwise the first that is not a number. A string
    // whose arithmetic fails errs the way the string library's
    // metamethods do.
    let bad = local("bad", i64());
    d.push(define(
        "zl_arith_meta",
        &[op, a, b],
        any(),
        vec![
            s.decl(event_name(op.e())),
            h.decl(call("zl_meta_of", vec![a.e(), s.e()], any())),
            when(
                is_nil(h.e()),
                vec![h.set(call("zl_meta_of", vec![b.e(), s.e()], any()))],
            ),
            when(
                not(is_nil(h.e())),
                vec![
                    metamethod_call_check(h.e(), operation_name(op.e())),
                    ret(call(
                        "zl_first",
                        vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                        any(),
                    )),
                ],
            ),
            when(
                eq(op.e(), int(OP_CONCAT)),
                vec![
                    bad.decl(if_expr(
                        and(not(is_nil(a.e())), is_text_like(a.e())),
                        int(OPERAND_RIGHT),
                        int(OPERAND_LEFT),
                    )),
                    x.decl(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
                    type_error(
                        concat(vec![
                            text("attempt to concatenate a "),
                            obj_type_name(x.e()),
                            text(" value"),
                        ]),
                        bad.e(),
                    ),
                    ret(nil()),
                ],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            nb.decl(call("zl_arith_operand", vec![b.e()], any())),
            bad.decl(if_expr(
                is_nil(na.e()),
                int(OPERAND_LEFT),
                int(OPERAND_RIGHT),
            )),
            x.decl(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
            // A bitwise operand is a number by type: two numbers, one
            // without an integer value, or the first that is not one.
            when(
                or(
                    and(ge(op.e(), int(OP_BAND)), le(op.e(), int(OP_SHR))),
                    eq(op.e(), int(OP_BNOT)),
                ),
                vec![
                    when(
                        and(is_number(a.e()), is_number(b.e())),
                        vec![
                            bad.set(if_expr(
                                has_int(a.e()),
                                int(OPERAND_RIGHT),
                                int(OPERAND_LEFT),
                            )),
                            type_error(
                                text("number has no integer representation"),
                                bitor(bad.e(), int(VARINFO_INSIDE)),
                            ),
                            ret(nil()),
                        ],
                    ),
                    bad.set(if_expr(
                        is_number(a.e()),
                        int(OPERAND_RIGHT),
                        int(OPERAND_LEFT),
                    )),
                    x.set(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
                    type_error(
                        concat(vec![
                            text("attempt to perform bitwise operation on a "),
                            obj_type_name(x.e()),
                            text(" value"),
                        ]),
                        bad.e(),
                    ),
                    ret(nil()),
                ],
            ),
            when(
                or(is_str(a.e()), is_str(b.e())),
                vec![lua_error(concat(vec![
                    text("attempt to "),
                    operation_name(op.e()),
                    text(" a '"),
                    type_name(a.e()),
                    text("' with a '"),
                    type_name(b.e()),
                    text("'"),
                ]))],
            ),
            type_error(
                concat(vec![
                    text("attempt to perform arithmetic on a "),
                    obj_type_name(x.e()),
                    text(" value"),
                ]),
                bad.e(),
            ),
            ret(nil()),
        ],
    ));
    // The integer a bitwise operand is, or -1 with `ok` false: an
    // integer, a float with an integral value, or a numeral string.
    // `out = the integer of x`, a number already; `scratch` holds the
    // float on the way. `fail` runs when it has no integer value.
    let to_int_stmts = |x: &Local, out: &Local, scratch: &Local, fail: Vec<Stmt>| {
        vec![if_(
            is_int_cat_of(x.e()),
            vec![out.set(get_i64(x.e()))],
            vec![
                scratch.set(get_f64(x.e())),
                when(
                    not(call("zl_float_is_int", vec![scratch.e()], boolean())),
                    fail,
                ),
                out.set(cast(scratch.e(), i64())),
            ],
        )]
    };

    // The initial value or step of a `for`, as the loop's number type;
    // `what` names which for the message.
    let what = kept("what", string());
    let for_number = |x: &Local, what: &Local| {
        vec![
            na.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("bad 'for' "),
                    what.e(),
                    text(" (number expected, got "),
                    obj_type_name(x.e()),
                    text(")"),
                ]))],
            ),
        ]
    };
    let mut st = for_number(a, &what);
    st.extend([
        when(is_int_cat_of(na.e()), vec![ret(get_i64(na.e()))]),
        f.decl(get_f64(na.e())),
        when(
            not(call("zl_float_is_int", vec![f.e()], boolean())),
            vec![lua_error(concat(vec![
                text("'for' "),
                what.e(),
                text(" must be an integer"),
            ]))],
        ),
        ret(cast(f.e(), i64())),
    ]);
    d.push(define("zl_for_int", &[a, &what], i64(), st));
    let mut st = for_number(a, &what);
    st.extend([
        when(
            is_int_cat_of(na.e()),
            vec![ret(cast(get_i64(na.e()), f64()))],
        ),
        ret(get_f64(na.e())),
    ]);
    d.push(define("zl_for_float", &[a, &what], f64(), st));
    // Whether a `for` whose start and step are dynamic values counts in
    // integers: both are integers, not numerals or floats.
    d.push(define(
        "zl_for_ints",
        &[a, b],
        boolean(),
        vec![
            when(or(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(false))]),
            ret(and(is_int_cat_of(a.e()), is_int_cat_of(b.e()))),
        ],
    ));
    // The integer limit of a `for` over integers with a float limit:
    // the last integer the loop may reach, so floored when the step
    // climbs and rounded up when it falls. Past the integers on the
    // step's own side it is clamped, which the loop never runs past;
    // on the other side the loop has no iteration, which the flag
    // says, since no integer limit could.
    let step = local("step", i64());
    let fl = local("fl", f64());
    d.push(define(
        "zl_for_limit_f",
        &[&fl, &step],
        i64(),
        vec![
            set_global(FOR_SKIP, bool(false)),
            f.decl(if_expr(
                lt(step.e(), int(0)),
                call("zl_ceil_f64", vec![fl.e()], f64()),
                call("floor", vec![fl.e()], f64()),
            )),
            when(
                ge(f.e(), float(9223372036854775808.0)),
                vec![
                    when(lt(step.e(), int(0)), vec![set_global(FOR_SKIP, bool(true))]),
                    ret(int(i64::MAX)),
                ],
            ),
            when(
                lt(f.e(), float(-9223372036854775808.0)),
                vec![
                    when(gt(step.e(), int(0)), vec![set_global(FOR_SKIP, bool(true))]),
                    ret(int(i64::MIN)),
                ],
            ),
            ret(cast(f.e(), i64())),
        ],
    ));
    // The same from a dynamic limit: a numeral read.
    d.push(define(
        "zl_for_limit",
        &[a, &step],
        i64(),
        vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("bad 'for' limit (number expected, got "),
                    obj_type_name(a.e()),
                    text(")"),
                ]))],
            ),
            when(
                is_int_cat_of(na.e()),
                vec![set_global(FOR_SKIP, bool(false)), ret(get_i64(na.e()))],
            ),
            ret(call(
                "zl_for_limit_f",
                vec![get_f64(na.e()), step.e()],
                i64(),
            )),
        ],
    ));
    // Typed helpers with Lua's rules, shared with the typed fast paths.
    // Division by -1 is a negation that wraps, never a trap.
    d.push(define(
        "zl_idiv_i64",
        &[ia, ib],
        i64(),
        vec![
            when(
                eq(ib.e(), int(0)),
                vec![lua_error(text("attempt to divide by zero"))],
            ),
            when(eq(ib.e(), int(-1)), vec![ret(sub(int(0), ia.e()))]),
            ret(call("zb_floordiv_i64", vec![ia.e(), ib.e()], i64())),
        ],
    ));
    d.push(define(
        "zl_mod_i64",
        &[ia, ib],
        i64(),
        vec![
            when(
                eq(ib.e(), int(0)),
                vec![lua_error(text("attempt to perform 'n%0'"))],
            ),
            when(eq(ib.e(), int(-1)), vec![ret(int(0))]),
            ret(call("zb_mod_i64", vec![ia.e(), ib.e()], i64())),
        ],
    ));
    d.push(define(
        "zl_idiv_f64",
        &[fa, fb],
        f64(),
        vec![ret(call("floor", vec![div(fa.e(), fb.e())], f64()))],
    ));
    // `a - floor(a/b)*b`, except that an infinite divisor keeps a finite
    // dividend of the other sign as itself plus the divisor's sign.
    d.push(define(
        "zl_mod_f64",
        &[fa, fb],
        f64(),
        vec![
            when(
                or(
                    eq(fb.e(), float(f64::INFINITY)),
                    eq(fb.e(), float(f64::NEG_INFINITY)),
                ),
                vec![
                    when(ne(fa.e(), fa.e()), vec![ret(fa.e())]),
                    when(
                        or(
                            eq(fa.e(), float(f64::INFINITY)),
                            eq(fa.e(), float(f64::NEG_INFINITY)),
                        ),
                        vec![ret(sub(fa.e(), fa.e()))],
                    ),
                    when(
                        or(
                            and(ge(fa.e(), float(0.0)), gt(fb.e(), float(0.0))),
                            and(le(fa.e(), float(0.0)), lt(fb.e(), float(0.0))),
                        ),
                        vec![ret(fa.e())],
                    ),
                    ret(fb.e()),
                ],
            ),
            ret(call("zb_mod_f64", vec![fa.e(), fb.e()], f64())),
        ],
    ));
    d.push(define(
        "zl_pow",
        &[fa, fb],
        f64(),
        vec![ret(call("pow", vec![fa.e(), fb.e()], f64()))],
    ));
    // Shifts: by 64 or more is zero, a negative count shifts the other
    // way, and a right shift is logical.
    d.push(define(
        "zl_shl_i64",
        &[ia, ib],
        i64(),
        vec![
            // A shift by 64 or more, either way, is zero; a negative
            // shift is the other direction.
            when(
                or(ge(ib.e(), int(64)), le(ib.e(), int(-64))),
                vec![ret(int(0))],
            ),
            when(
                lt(ib.e(), int(0)),
                vec![ret(call(
                    "zl_shr_i64",
                    vec![ia.e(), sub(int(0), ib.e())],
                    i64(),
                ))],
            ),
            ret(shl(ia.e(), ib.e())),
        ],
    ));
    // The shift right is logical: the bits the arithmetic shift fills
    // with the sign are masked off.
    d.push(define(
        "zl_shr_i64",
        &[ia, ib],
        i64(),
        vec![
            when(
                or(ge(ib.e(), int(64)), le(ib.e(), int(-64))),
                vec![ret(int(0))],
            ),
            when(
                lt(ib.e(), int(0)),
                vec![ret(call(
                    "zl_shl_i64",
                    vec![ia.e(), sub(int(0), ib.e())],
                    i64(),
                ))],
            ),
            when(eq(ib.e(), int(0)), vec![ret(ia.e())]),
            ret(bitand(
                shr(ia.e(), ib.e()),
                sub(shl(int(1), sub(int(64), ib.e())), int(1)),
            )),
        ],
    ));
    // The operator on two integers, boxed.
    let int_op = |op: &Local, a: Expr, b: Expr| -> Vec<Stmt> {
        let pick = |code: i64, v: Expr| when(eq(op.e(), int(code)), vec![ret(box_i64(v))]);
        vec![
            pick(OP_ADD, add(a.clone(), b.clone())),
            pick(OP_SUB, sub(a.clone(), b.clone())),
            pick(OP_MUL, mul(a.clone(), b.clone())),
            pick(
                OP_IDIV,
                call("zl_idiv_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(
                OP_MOD,
                call("zl_mod_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(OP_BAND, bitand(a.clone(), b.clone())),
            pick(OP_BOR, bitor(a.clone(), b.clone())),
            pick(OP_BXOR, bitxor(a.clone(), b.clone())),
            pick(
                OP_SHL,
                call("zl_shl_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(OP_SHR, call("zl_shr_i64", vec![a, b], i64())),
        ]
    };
    let float_op = |op: &Local, a: Expr, b: Expr| -> Vec<Stmt> {
        let pick = |code: i64, v: Expr| when(eq(op.e(), int(code)), vec![ret(box_f64(v))]);
        vec![
            pick(OP_ADD, add(a.clone(), b.clone())),
            pick(OP_SUB, sub(a.clone(), b.clone())),
            pick(OP_MUL, mul(a.clone(), b.clone())),
            pick(OP_DIV, div(a.clone(), b.clone())),
            pick(OP_POW, call("zl_pow", vec![a.clone(), b.clone()], f64())),
            pick(
                OP_IDIV,
                call("zl_idiv_f64", vec![a.clone(), b.clone()], f64()),
            ),
            pick(OP_MOD, call("zl_mod_f64", vec![a, b], f64())),
        ]
    };
    let as_f64 = |x: Expr| {
        if_expr(
            is_int_cat_of(x.clone()),
            cast(get_i64(x.clone()), f64()),
            get_f64(x),
        )
    };
    // `a op b` on dynamic values. Tables go to their metamethods first;
    // otherwise both operands are numbers, integers when both are and
    // the operator keeps integers.
    let na = kept("na", any());
    let nb = kept("nb", any());
    d.push(define("zl_arith", &[op, a, b], any(), {
        let mut st = vec![
            when(
                eq(op.e(), int(OP_CONCAT)),
                vec![ret(call("zl_concat", vec![a.e(), b.e()], any()))],
            ),
            when(
                or(is_table(a.e()), is_table(b.e())),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![op.e(), a.e(), b.e()],
                    any(),
                ))],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            nb.decl(call("zl_arith_operand", vec![b.e()], any())),
            when(
                or(is_nil(na.e()), is_nil(nb.e())),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![op.e(), a.e(), b.e()],
                    any(),
                ))],
            ),
        ];
        // Bitwise operators convert floats with integral values, never
        // strings: those are the string metatable's to handle.
        let meta = || ret(call("zl_arith_meta", vec![op.e(), a.e(), b.e()], any()));
        let mut bitwise = vec![when(
            or(eq(category(a.e()), int(STR)), eq(category(b.e()), int(STR))),
            vec![meta()],
        )];
        bitwise.extend(to_int_stmts(&na, ia, fa, vec![meta()]));
        bitwise.extend(to_int_stmts(&nb, ib, fb, vec![meta()]));
        bitwise.extend(int_op(op, ia.e(), ib.e()));
        st.push(ia.decl(int(0)));
        st.push(ib.decl(int(0)));
        st.push(fa.decl(float(0.0)));
        st.push(fb.decl(float(0.0)));
        st.push(when(
            and(ge(op.e(), int(OP_BAND)), le(op.e(), int(OP_SHR))),
            bitwise,
        ));
        let mut ints = vec![ia.set(get_i64(na.e())), ib.set(get_i64(nb.e()))];
        ints.extend(int_op(op, ia.e(), ib.e()));
        st.push(when(
            and(
                and(is_int_cat_of(na.e()), is_int_cat_of(nb.e())),
                and(ne(op.e(), int(OP_DIV)), ne(op.e(), int(OP_POW))),
            ),
            ints,
        ));
        st.push(fa.set(as_f64(na.e())));
        st.push(fb.set(as_f64(nb.e())));
        st.extend(float_op(op, fa.e(), fb.e()));
        st.push(ret(nil()));
        st
    }));
    // Unary minus and bitwise not.
    d.push(define(
        "zl_unm",
        &[a],
        any(),
        vec![
            when(
                is_table(a.e()),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![int(OP_UNM), a.e(), a.e()],
                    any(),
                ))],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![int(OP_UNM), a.e(), a.e()],
                    any(),
                ))],
            ),
            when(
                is_int_cat_of(na.e()),
                vec![ret(box_i64(sub(int(0), get_i64(na.e()))))],
            ),
            ret(box_f64(sub(float(0.0), get_f64(na.e())))),
        ],
    ));
    d.push(define("zl_bnot", &[a], any(), {
        let meta = || {
            ret(call(
                "zl_arith_meta",
                vec![int(OP_BNOT), a.e(), a.e()],
                any(),
            ))
        };
        let mut st = vec![
            when(or(is_table(a.e()), is_str(a.e())), vec![meta()]),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(is_nil(na.e()), vec![meta()]),
            ia.decl(int(0)),
            fa.decl(float(0.0)),
        ];
        st.extend(to_int_stmts(&na, ia, fa, vec![meta()]));
        st.push(ret(box_i64(bitxor(ia.e(), int(-1)))));
        st
    }));
    // The integer a dynamic value holds, for a typed bitwise operand.
    d.push(define("zl_toint", &[a], i64(), {
        let mut st = vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("attempt to perform bitwise operation on a "),
                    obj_type_name(a.e()),
                    text(" value"),
                ]))],
            ),
            ia.decl(int(0)),
            fa.decl(float(0.0)),
        ];
        st.extend(to_int_stmts(
            &na,
            ia,
            fa,
            vec![lua_error(text("number has no integer representation"))],
        ));
        st.push(ret(ia.e()));
        st
    }));
    // The float a dynamic value holds, for a typed float operand.
    d.push(define(
        "zl_tofloat",
        &[a],
        f64(),
        vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("attempt to perform arithmetic on a "),
                    obj_type_name(a.e()),
                    text(" value"),
                ]))],
            ),
            ret(as_f64(na.e())),
        ],
    ));
}
