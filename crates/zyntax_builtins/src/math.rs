//! Mathematics over `f64` and `i64`: what a language's `math` module
//! offers. Elementary functions come from the math plugin; rounding,
//! `gcd` and `factorial` are written here.

use crate::build::*;

pub(crate) fn declarations() -> Vec<Decl> {
    let x = local("x", f64());
    let y = local("y", f64());
    let mut d = Vec::new();

    // One argument in, one out, from the plugin.
    for (name, symbol) in [
        ("exp", "$Math$exp"),
        ("log", "$Math$log"),
        ("log2", "$Math$log2"),
        ("log10", "$Math$log10"),
        ("sin", "$Math$sin"),
        ("cos", "$Math$cos"),
        ("tan", "$Math$tan"),
        ("asin", "$Math$asin"),
        ("acos", "$Math$acos"),
        ("atan", "$Math$atan"),
        ("sinh", "$Math$sinh"),
        ("cosh", "$Math$cosh"),
        ("tanh", "$Math$tanh"),
    ] {
        d.push(extern_fn(
            &format!("zb_math_{name}"),
            &[("x", f64())],
            f64(),
            Some(symbol),
        ));
    }
    for (name, symbol) in [
        ("atan2", "$Math$atan2"),
        ("hypot", "$Math$hypot"),
        ("fmod", "$Math$fmod"),
        ("copysign", "$Math$copysign"),
    ] {
        d.push(extern_fn(
            &format!("zb_math_{name}"),
            &[("x", f64()), ("y", f64())],
            f64(),
            Some(symbol),
        ));
    }

    // The intrinsics the compiler has.
    d.push(extern_fn("sqrt", &[("x", f64())], f64(), None));
    d.push(define(
        "zb_math_sqrt",
        &[&x],
        f64(),
        vec![
            when(
                lt(x.e(), float(0.0)),
                vec![fatal("ValueError", text("math domain error"))],
            ),
            ret(call("sqrt", vec![x.e()], f64())),
        ],
    ));
    d.push(define(
        "zb_math_pow",
        &[&x, &y],
        f64(),
        vec![ret(call("pow", vec![x.e(), y.e()], f64()))],
    ));
    d.push(define(
        "zb_math_fabs",
        &[&x],
        f64(),
        vec![
            when(lt(x.e(), float(0.0)), vec![ret(sub(float(0.0), x.e()))]),
            ret(x.e()),
        ],
    ));
    d.push(define(
        "zb_math_floor",
        &[&x],
        i64(),
        vec![ret(cast(call("floor", vec![x.e()], f64()), i64()))],
    ));
    d.push(define(
        "zb_math_ceil",
        &[&x],
        i64(),
        vec![ret(cast(
            sub(
                float(0.0),
                call("floor", vec![sub(float(0.0), x.e())], f64()),
            ),
            i64(),
        ))],
    ));
    d.push(define(
        "zb_math_trunc",
        &[&x],
        i64(),
        vec![ret(cast(x.e(), i64()))],
    ));
    d.push(define(
        "zb_math_isnan",
        &[&x],
        boolean(),
        vec![ret(ne(x.e(), x.e()))],
    ));
    d.push(define(
        "zb_math_isinf",
        &[&x],
        boolean(),
        vec![ret(or(
            eq(x.e(), float(f64::INFINITY)),
            eq(x.e(), float(f64::NEG_INFINITY)),
        ))],
    ));
    d.push(define(
        "zb_math_isfinite",
        &[&x],
        boolean(),
        vec![ret(and(
            eq(x.e(), x.e()),
            and(
                ne(x.e(), float(f64::INFINITY)),
                ne(x.e(), float(f64::NEG_INFINITY)),
            ),
        ))],
    ));
    d.push(define(
        "zb_math_log_base",
        &[&x, &y],
        f64(),
        vec![ret(div(
            call("zb_math_log", vec![x.e()], f64()),
            call("zb_math_log", vec![y.e()], f64()),
        ))],
    ));
    d.push(define(
        "zb_math_degrees",
        &[&x],
        f64(),
        vec![ret(mul(x.e(), float(180.0 / std::f64::consts::PI)))],
    ));
    d.push(define(
        "zb_math_radians",
        &[&x],
        f64(),
        vec![ret(mul(x.e(), float(std::f64::consts::PI / 180.0)))],
    ));

    // Integers.
    let a = local("a", i64());
    let b = local("b", i64());
    let p = local("p", i64());
    let q = local("q", i64());
    let t = local("t", i64());
    d.push(define(
        "zb_math_gcd",
        &[&a, &b],
        i64(),
        vec![
            p.decl(if_expr(lt(a.e(), int(0)), sub(int(0), a.e()), a.e())),
            q.decl(if_expr(lt(b.e(), int(0)), sub(int(0), b.e()), b.e())),
            while_(
                ne(q.e(), int(0)),
                vec![t.decl(rem(p.e(), q.e())), p.set(q.e()), q.set(t.e())],
            ),
            ret(p.e()),
        ],
    ));
    let n = local("n", i64());
    let acc = local("acc", i64());
    let i = local("i", i64());
    d.push(define("zb_math_factorial", &[&n], i64(), {
        let mut s = vec![
            when(
                lt(n.e(), int(0)),
                vec![fatal(
                    "ValueError",
                    text("factorial() not defined for negative values"),
                )],
            ),
            acc.decl(int(1)),
        ];
        s.extend(for_range(
            &i,
            int(2),
            add(n.e(), int(1)),
            vec![acc.set(mul(acc.e(), i.e()))],
        ));
        s.push(ret(acc.e()));
        s
    }));
    d
}
