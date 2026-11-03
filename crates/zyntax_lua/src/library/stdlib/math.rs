//! Math declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        f,
        g,
        i,
        j,
        n,
        x,
        y,
        ..
    } = context;
    // ─── math ───────────────────────────────────────────────────
    let number_arg = |x: &Local, what: &str| {
        vec![
            y.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(
                is_nil(y.e()),
                vec![lua_error(concat(vec![
                    text(&format!(
                        "bad argument #1 to '{what}' (number expected, got "
                    )),
                    if_expr(is_nil(x.e()), text("no value"), arg_type_name(x.e())),
                    text(")"),
                ]))],
            ),
        ]
    };
    let is_int_box = |x: Expr| {
        let c = category(x);
        or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
    };
    // A float with an integral value that fits becomes an integer.
    let int_if_fits = |f: Expr| {
        if_expr(
            call("zl_float_is_int", vec![f.clone()], boolean()),
            box_i64(cast(f.clone(), i64())),
            box_f64(f),
        )
    };
    d.push(define(
        "zl_ceil_f64",
        &[f],
        f64(),
        vec![ret(sub(
            float(0.0),
            call("floor", vec![sub(float(0.0), f.e())], f64()),
        ))],
    ));
    for (name, what, op) in [
        ("zl_math_floor", "floor", "floor"),
        ("zl_math_ceil", "ceil", "zl_ceil_f64"),
    ] {
        let mut st = number_arg(x, what);
        st.push(when(is_int_box(y.e()), vec![ret(y.e())]));
        st.push(ret(int_if_fits(call(op, vec![get_f64(y.e())], f64()))));
        d.push(define(name, &[x], any(), st));
    }
    // The absolute value the compiler has an instruction for.
    d.push(extern_fn("abs", &[("x", f64())], f64(), None));
    // `math.fmod` on two integers: the remainder rounded toward zero;
    // by -1 it is zero outright, since the least integer's overflows.
    let ia = local("ia", i64());
    let ib = local("ib", i64());
    d.push(define(
        "zl_fmod_i64",
        &[&ia, &ib],
        i64(),
        vec![
            when(
                eq(ib.e(), int(0)),
                vec![lua_error(text("bad argument #2 to 'fmod' (zero)"))],
            ),
            when(eq(ib.e(), int(-1)), vec![ret(int(0))]),
            ret(rem(ia.e(), ib.e())),
        ],
    ));
    d.push(define("zl_math_abs", &[x], any(), {
        let mut st = number_arg(x, "abs");
        st.push(when(
            is_int_box(y.e()),
            vec![ret(if_expr(
                lt(get_i64(y.e()), int(0)),
                box_i64(sub(int(0), get_i64(y.e()))),
                y.e(),
            ))],
        ));
        st.push(ret(box_f64(call("abs", vec![get_f64(y.e())], f64()))));
        st
    }));
    for (name, shared) in [
        ("zl_math_sqrt", "sqrt"),
        ("zl_math_exp", "zb_math_exp"),
        ("zl_math_sin", "zb_math_sin"),
        ("zl_math_cos", "zb_math_cos"),
        ("zl_math_tan", "zb_math_tan"),
        ("zl_math_asin", "zb_math_asin"),
        ("zl_math_acos", "zb_math_acos"),
    ] {
        d.push(define(
            name,
            &[f],
            f64(),
            vec![ret(call(shared, vec![f.e()], f64()))],
        ));
    }
    d.push(define(
        "zl_math_deg",
        &[f],
        f64(),
        vec![ret(mul(f.e(), float(180.0 / std::f64::consts::PI)))],
    ));
    d.push(define(
        "zl_math_rad",
        &[f],
        f64(),
        vec![ret(mul(f.e(), float(std::f64::consts::PI / 180.0)))],
    ));
    d.push(define(
        "zl_math_atan",
        &[f, x],
        f64(),
        vec![
            g.decl(float(1.0)),
            when(
                not(is_nil(x.e())),
                vec![g.set(call("zl_arg_float", vec![x.e(), bad_arg(2, "atan")], f64()))],
            ),
            ret(call("zb_math_atan2", vec![f.e(), g.e()], f64())),
        ],
    ));
    d.push(define(
        "zl_math_log",
        &[f, x],
        f64(),
        vec![
            when(
                is_nil(x.e()),
                vec![ret(call("zb_math_log", vec![f.e()], f64()))],
            ),
            g.decl(call("zl_arg_float", vec![x.e(), bad_arg(2, "log")], f64())),
            when(
                eq(g.e(), float(2.0)),
                vec![ret(call("zb_math_log2", vec![f.e()], f64()))],
            ),
            when(
                eq(g.e(), float(10.0)),
                vec![ret(call("zb_math_log10", vec![f.e()], f64()))],
            ),
            ret(div(
                call("zb_math_log", vec![f.e()], f64()),
                call("zb_math_log", vec![g.e()], f64()),
            )),
        ],
    ));
    // `math.fmod`: the remainder rounded toward zero.
    d.push(define(
        "zl_math_fmod",
        &[x, y],
        any(),
        vec![
            when(
                and(is_int_box(x.e()), is_int_box(y.e())),
                vec![ret(box_i64(call(
                    "zl_fmod_i64",
                    vec![get_i64(x.e()), get_i64(y.e())],
                    i64(),
                )))],
            ),
            ret(box_f64(rem(
                call("zl_arg_float", vec![x.e(), bad_arg(1, "fmod")], f64()),
                call("zl_arg_float", vec![y.e(), bad_arg(2, "fmod")], f64()),
            ))),
        ],
    ));
    // `math.modf`: the integral part (an integer when it fits) and the
    // fraction, which is zero for an infinity.
    let whole = kept("whole", any());
    let fraction = local("fraction", f64());
    d.push(define(
        "zl_math_modf",
        &[f],
        any(),
        vec![
            g.decl(float(0.0)),
            if_(
                ge(f.e(), float(0.0)),
                vec![g.set(call("floor", vec![f.e()], f64()))],
                vec![g.set(call("zl_ceil_f64", vec![f.e()], f64()))],
            ),
            whole.decl(nil()),
            if_(
                call("zl_float_is_int", vec![g.e()], boolean()),
                vec![whole.set(box_i64(cast(g.e(), i64())))],
                vec![whole.set(box_f64(g.e()))],
            ),
            fraction.decl(sub(f.e(), g.e())),
            when(
                or(
                    eq(f.e(), float(f64::INFINITY)),
                    eq(f.e(), float(f64::NEG_INFINITY)),
                ),
                vec![fraction.set(float(0.0))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![whole.e(), box_f64(fraction.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    for (name, what, pick) in [
        ("zl_math_max", "max", "zl_lt"),
        ("zl_math_min", "min", "zl_lt"),
    ] {
        let best = kept("best", any());
        let better = if name == "zl_math_max" {
            call(pick, vec![best.e(), at(args.e(), i.e())], boolean())
        } else {
            call(pick, vec![at(args.e(), i.e()), best.e()], boolean())
        };
        d.push(define(
            name,
            &[args],
            any(),
            vec![
                // The values are compared as `<` compares them, whatever
                // they are; one alone is the answer.
                when(
                    eq(len(args.e()), int(0)),
                    vec![lua_error(text(&format!(
                        "bad argument #1 to '{what}' (value expected)"
                    )))],
                ),
                best.decl(at(args.e(), int(0))),
                i.decl(int(1)),
                while_(
                    lt(i.e(), len(args.e())),
                    vec![
                        when(better, vec![best.set(at(args.e(), i.e()))]),
                        i.add_assign(int(1)),
                    ],
                ),
                ret(best.e()),
            ],
        ));
    }
    d.push(define(
        "zl_math_tointeger",
        &[x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            when(is_int_box(x.e()), vec![ret(x.e())]),
            when(
                eq(category(x.e()), int(FLOAT)),
                vec![
                    f.decl(get_f64(x.e())),
                    when(
                        call("zl_float_is_int", vec![f.e()], boolean()),
                        vec![ret(box_i64(cast(f.e(), i64())))],
                    ),
                    ret(nil()),
                ],
            ),
            when(
                eq(category(x.e()), int(STR)),
                vec![ret(call(
                    "zl_math_tointeger",
                    vec![call("zl_str_to_number", vec![get_str(x.e())], any())],
                    any(),
                ))],
            ),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_math_type",
        &[x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            when(is_int_box(x.e()), vec![ret(box_str(text("integer")))]),
            when(
                eq(category(x.e()), int(FLOAT)),
                vec![ret(box_str(text("float")))],
            ),
            ret(nil()),
        ],
    ));
    let lo = local("lo", i64());
    let hi = local("hi", i64());
    d.push(define(
        "zl_math_random",
        &[args],
        any(),
        vec![
            n.decl(len(args.e())),
            when(
                eq(n.e(), int(0)),
                vec![ret(box_f64(call("zl_random_float", vec![], f64())))],
            ),
            when(
                gt(n.e(), int(2)),
                vec![lua_error(text("wrong number of arguments"))],
            ),
            lo.decl(int(1)),
            hi.decl(call(
                "zl_arg_int",
                vec![at(args.e(), int(0)), bad_arg(1, "random")],
                i64(),
            )),
            // `random(0)`: a whole draw.
            when(
                and(eq(n.e(), int(1)), eq(hi.e(), int(0))),
                vec![ret(box_i64(call("zl_random_next", vec![], i64())))],
            ),
            when(
                ge(n.e(), int(2)),
                vec![
                    lo.set(hi.e()),
                    hi.set(call(
                        "zl_arg_int",
                        vec![at(args.e(), int(1)), bad_arg(2, "random")],
                        i64(),
                    )),
                ],
            ),
            when(
                gt(lo.e(), hi.e()),
                vec![lua_error(text(
                    "bad argument #1 to 'random' (interval is empty)",
                ))],
            ),
            ret(box_i64(call("zl_random_int", vec![lo.e(), hi.e()], i64()))),
        ],
    ));
    // `math.randomseed([n1 [, n2]])`: the two seed numbers are
    // answered, so a run can be repeated. A float seed is taken by its
    // integer value; nothing given seeds from the clock.
    d.push(define(
        "zl_math_randomseed",
        &[args],
        any(),
        vec![
            i.decl(int(0)),
            j.decl(int(0)),
            if_(
                eq(len(args.e()), int(0)),
                vec![
                    i.set(call("zl_random_seed_now", vec![], i64())),
                    j.set(call("zl_random_seed_pid", vec![], i64())),
                ],
                vec![
                    x.decl(at(args.e(), int(0))),
                    i.set(if_expr(
                        and(not(is_nil(x.e())), is_int_box(x.e())),
                        get_i64(x.e()),
                        cast(
                            call("zl_arg_float", vec![x.e(), bad_arg(1, "randomseed")], f64()),
                            i64(),
                        ),
                    )),
                    when(
                        gt(len(args.e()), int(1)),
                        vec![j.set(call(
                            "zl_arg_int",
                            vec![at(args.e(), int(1)), bad_arg(2, "randomseed")],
                            i64(),
                        ))],
                    ),
                    expr(call("zl_random_seed", vec![i.e(), j.e()], unit())),
                ],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(vec![box_i64(i.e()), box_i64(j.e())], anys.clone())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_math_ult",
        &[i, j],
        boolean(),
        vec![ret(lt(cast(i.e(), u64()), cast(j.e(), u64())))],
    ));
}
