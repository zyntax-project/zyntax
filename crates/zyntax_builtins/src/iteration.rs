//! Iteration over sequences of dynamic values: pairing (`enumerate`,
//! `zip`), applying a function value (`map`, `filter`), and quantifying
//! (`any`, `all`). Each takes and returns a `List<Any>`; a frontend with
//! a typed list boxes it first and unboxes what comes back. An eager list
//! stands in for the lazy iterator: the two are told apart only by an
//! infinite or side-effecting source.

use crate::build::*;
use crate::list_of;
use zyntax_typed_ast::TypeId;

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let empty = || list(Vec::new(), anys.clone());
    let pair = |a: Expr, b: Expr| call("zb_box_tuple", vec![list(vec![a, b], anys.clone())], any());
    let push = |out: &Local, v: Expr| expr(mcall(out.e(), "push", vec![v], unit()));
    let box_int = |v: Expr| call("zb_box_i64", vec![v], any());
    let truthy = |v: Expr| call("zb_any_truthy", vec![v], boolean());
    let call1 = |f: &Local, v: Expr| call("zb_call_1", vec![f.e(), v], any());

    let xs = local("xs", anys.clone());
    let ys = local("ys", anys.clone());
    let zs = local("zs", anys.clone());
    let out = local("out", anys.clone());
    let f = local("f", any());
    let i = local("i", i64());
    let n = local("n", i64());
    let start = local("start", i64());
    let len = |xs: &Local| mcall(xs.e(), "len", vec![], i64());
    let at = |xs: &Local, i: Expr| idx(xs.e(), i, any());
    let mut d = Vec::new();

    let env = local("env", anys.clone());
    let packed = local("packed", any());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let bound = |at: i64| call("zb_any_as_i64", vec![idx(xs.e(), int(at), any())], i64());
    d.push(define(
        "zb_range_call",
        &[&env, &packed],
        any(),
        vec![
            xs.decl(call("zb_list_unbox_any", vec![packed.e()], anys.clone())),
            n.decl(len(&xs)),
            when(
                lt(n.e(), int(1)),
                vec![fatal(
                    "TypeError",
                    text("range expected at least 1 argument, got 0"),
                )],
            ),
            when(
                gt(n.e(), int(3)),
                vec![fatal(
                    "TypeError",
                    text("range expected at most 3 arguments"),
                )],
            ),
            start.decl(int(0)),
            stop.decl(bound(0)),
            step.decl(int(1)),
            when(
                gt(n.e(), int(1)),
                vec![start.set(stop.e()), stop.set(bound(1))],
            ),
            when(eq(n.e(), int(3)), vec![step.set(bound(2))]),
            when(
                eq(step.e(), int(0)),
                vec![fatal("ValueError", text("range() arg 3 must not be zero"))],
            ),
            ret(call(
                "zb_list_box_i64",
                vec![call(
                    "zb_list_range",
                    vec![start.e(), stop.e(), step.e()],
                    list_of(list_type, i64()),
                )],
                any(),
            )),
        ],
    ));

    d.push(define("zb_list_enumerate", &[&xs, &start], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![push(
                &out,
                pair(box_int(add(start.e(), i.e())), at(&xs, i.e())),
            )],
        ));
        s.push(ret(out.e()));
        s
    }));

    // The shortest input decides the length.
    let shorter = |a: Expr, b: Expr| if_expr(lt(a.clone(), b.clone()), a, b);
    d.push(define("zb_list_zip2", &[&xs, &ys], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(shorter(len(&xs), len(&ys)))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![push(&out, pair(at(&xs, i.e()), at(&ys, i.e())))],
        ));
        s.push(ret(out.e()));
        s
    }));
    d.push(define("zb_list_zip3", &[&xs, &ys, &zs], anys.clone(), {
        let mut s = vec![
            out.decl(empty()),
            n.decl(shorter(shorter(len(&xs), len(&ys)), len(&zs))),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![push(
                &out,
                call(
                    "zb_box_tuple",
                    vec![list(
                        vec![at(&xs, i.e()), at(&ys, i.e()), at(&zs, i.e())],
                        anys.clone(),
                    )],
                    any(),
                ),
            )],
        ));
        s.push(ret(out.e()));
        s
    }));

    d.push(define("zb_list_map1", &[&f, &xs], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![push(&out, call1(&f, at(&xs, i.e())))],
        ));
        s.push(ret(out.e()));
        s
    }));
    d.push(define("zb_list_map2", &[&f, &xs, &ys], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(shorter(len(&xs), len(&ys)))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![push(
                &out,
                call(
                    "zb_call_2",
                    vec![f.e(), at(&xs, i.e()), at(&ys, i.e())],
                    any(),
                ),
            )],
        ));
        s.push(ret(out.e()));
        s
    }));

    let x = local("x", any());
    d.push(define("zb_list_filter", &[&f, &xs], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                x.decl(at(&xs, i.e())),
                when(truthy(call1(&f, x.e())), vec![push(&out, x.e())]),
            ],
        ));
        s.push(ret(out.e()));
        s
    }));
    d.push(define("zb_list_filter_truthy", &[&xs], anys.clone(), {
        let mut s = vec![out.decl(empty()), n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                x.decl(at(&xs, i.e())),
                when(truthy(x.e()), vec![push(&out, x.e())]),
            ],
        ));
        s.push(ret(out.e()));
        s
    }));

    d.push(define("zb_list_any", &[&xs], boolean(), {
        let mut s = vec![n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(truthy(at(&xs, i.e())), vec![ret(bool(true))])],
        ));
        s.push(ret(bool(false)));
        s
    }));
    d.push(define("zb_list_all", &[&xs], boolean(), {
        let mut s = vec![n.decl(len(&xs))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(not(truthy(at(&xs, i.e()))), vec![ret(bool(false))])],
        ));
        s.push(ret(bool(true)));
        s
    }));

    // A dict from a list of pairs, later pairs winning.
    let items = local("items", anys.clone());
    let kv = local("kv", anys.clone());
    d.push(define("zb_dict_from_tuples", &[&items], anys.clone(), {
        let mut s = vec![
            out.decl(call("zb_dict_new", vec![], anys.clone())),
            n.decl(len(&items)),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                kv.decl(call(
                    "zb_unbox_tuple",
                    vec![at(&items, i.e())],
                    anys.clone(),
                )),
                when(
                    ne(len(&kv), int(2)),
                    vec![fatal(
                        "ValueError",
                        text("dictionary update sequence element is not a pair"),
                    )],
                ),
                expr(call(
                    "zb_dict_set",
                    vec![out.e(), at(&kv, int(0)), at(&kv, int(1))],
                    unit(),
                )),
            ],
        ));
        s.push(ret(out.e()));
        s
    }));

    // pow(base, exp, m) by squaring; the result takes the sign of m.
    let base = local("base", i64());
    let exp = local("exp", i64());
    let m = local("m", i64());
    let acc = local("acc", i64());
    let b = local("b", i64());
    let e = local("e", i64());
    d.push(define(
        "zb_int_pow_mod",
        &[&base, &exp, &m],
        i64(),
        vec![
            when(
                eq(m.e(), int(0)),
                vec![fatal("ValueError", text("pow() 3rd argument cannot be 0"))],
            ),
            when(
                lt(exp.e(), int(0)),
                vec![fatal(
                    "ValueError",
                    text("pow() with a negative exponent and a modulus is not supported"),
                )],
            ),
            acc.decl(rem(int(1), m.e())),
            b.decl(rem(base.e(), m.e())),
            e.decl(exp.e()),
            while_(
                gt(e.e(), int(0)),
                vec![
                    when(
                        eq(rem(e.e(), int(2)), int(1)),
                        vec![acc.set(rem(mul(acc.e(), b.e()), m.e()))],
                    ),
                    b.set(rem(mul(b.e(), b.e()), m.e())),
                    e.set(div(e.e(), int(2))),
                ],
            ),
            // `rem` follows the dividend; Python follows the divisor.
            when(
                and(
                    ne(acc.e(), int(0)),
                    ne(lt(acc.e(), int(0)), lt(m.e(), int(0))),
                ),
                vec![acc.set(add(acc.e(), m.e()))],
            ),
            ret(acc.e()),
        ],
    ));
    d
}
