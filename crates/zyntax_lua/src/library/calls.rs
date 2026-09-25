//! Calls through function values, and multiple values.
//!
//! A function value is the shared library's record: code, arity word,
//! then the variables it shares with the scope that made it. Its code
//! takes the record and exactly as many dynamic arguments as the arity
//! says, or the record and one boxed list when the function is
//! variadic. A call adjusts as Lua does: missing arguments are nil,
//! extra ones are dropped. A table with a `__call` metamethod is
//! called through it.
//!
//! Several values travel as one dynamic value: a list under the tuple
//! tag. Exactly one value travels as itself, so a call that returns one
//! value costs no list; `zl_first` and `zl_values` read either form.

use super::*;
use zyntax_builtins::functions::{MAX_CALL_ARITY, VARIADIC_ARITY, code_type};

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let f = kept("f", any());
    let x = kept("x", any());
    let h = kept("h", any());
    let rec = borrowed("rec", anys.clone());
    let arity = local("arity", i64());
    let y = local("y", any());
    let most = local("most", i64());
    let xs = kept("xs", anys.clone());
    let out = borrowed("out", anys.clone());
    let args = kept("args", anys.clone());
    let i = local("i", i64());
    let n = local("n", i64());
    let mut d = Vec::new();

    let is_tuple = |x: Expr| {
        and(
            ne(x.clone(), nil()),
            eq(tag_of(x), int(zyntax_builtins::TUPLE_TAG)),
        )
    };
    let tuple_items = |x: Expr| call("zb_unbox_tuple_raw", vec![x], anys.clone());
    let fp_name = |n: usize| format!("zb_unbox_fnptr_raw_{n}");

    // ─── several values ─────────────────────────────────────────
    d.push(define(
        "zl_first",
        &[&x],
        any(),
        vec![
            when(
                is_tuple(x.e()),
                vec![
                    xs.decl(tuple_items(x.e())),
                    when(eq(len(xs.e()), int(0)), vec![ret(nil())]),
                    ret(at(xs.e(), int(0))),
                ],
            ),
            ret(x.e()),
        ],
    ));
    d.push(define(
        "zl_values",
        &[&x],
        anys.clone(),
        vec![
            when(is_tuple(x.e()), vec![ret(tuple_items(x.e()))]),
            ret(list(vec![x.e()], anys.clone())),
        ],
    ));
    // The values of `x` appended to `out`.
    d.push(define(
        "zl_append_values",
        &[&out, &x],
        unit(),
        vec![
            if_(
                is_tuple(x.e()),
                vec![expr(call(
                    "zb_list_extend_any",
                    vec![out.e(), tuple_items(x.e())],
                    unit(),
                ))],
                vec![push(out.e(), x.e())],
            ),
            ret_void(),
        ],
    ));
    // No values at all, as one: what a function that returns nothing
    // returns to a dynamic caller.
    d.push(define(
        "zl_none",
        &[],
        any(),
        vec![
            xs.decl(list(vec![], anys.clone())),
            ret(call("zb_box_tuple", vec![xs.e()], any())),
        ],
    ));
    // Several values as one: exactly one is itself.
    d.push(define(
        "zl_pack",
        &[&xs],
        any(),
        vec![
            when(eq(len(xs.e()), int(1)), vec![ret(at(xs.e(), int(0)))]),
            ret(call("zb_box_tuple", vec![xs.e()], any())),
        ],
    ));
    // Value `i` (1-based) of a list, nil past its end.
    d.push(define(
        "zl_value_at",
        &[&xs, &i],
        any(),
        vec![
            when(
                or(lt(i.e(), int(1)), gt(i.e(), len(xs.e()))),
                vec![ret(nil())],
            ),
            ret(at(xs.e(), sub(i.e(), int(1)))),
        ],
    ));
    // The items of a list from `i` to `j` (0-based, `j` excluded).
    let j = local("j", i64());
    let k = local("k", i64());
    d.push(define(
        "zl_slice",
        &[&xs, &i, &j],
        anys.clone(),
        vec![
            out.decl(list(vec![], anys.clone())),
            k.decl(i.e()),
            while_(
                lt(k.e(), j.e()),
                vec![push(out.e(), at(xs.e(), k.e())), k.add_assign(int(1))],
            ),
            ret(out.e()),
        ],
    ));
    // The values from `i` (1-based) on, as one value.
    d.push(define(
        "zl_values_from",
        &[&xs, &i],
        any(),
        vec![
            n.decl(len(xs.e())),
            when(
                gt(i.e(), n.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(vec![], anys.clone())],
                    any(),
                ))],
            ),
            when(eq(i.e(), n.e()), vec![ret(at(xs.e(), sub(i.e(), int(1))))]),
            ret(call(
                "zb_box_tuple",
                vec![call(
                    "zl_slice",
                    vec![xs.e(), sub(i.e(), int(1)), n.e()],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));

    // ─── the record of a callee ─────────────────────────────────
    // The record `f` is, or the `__call` handler's with `f` put in
    // front of `args`; `args` is edited in place for the caller.
    let not_callable = |f: &Local| {
        type_error(
            concat(vec![
                text("attempt to call a "),
                type_name(f.e()),
                text(" value"),
            ]),
            int(OPERAND_LEFT | VARINFO_CALL),
        )
    };
    d.push(define(
        "zl_callee",
        &[&f, &args],
        anys.clone(),
        vec![
            when(
                is_func(f.e()),
                vec![ret(call(
                    "zb_unbox_list_raw_any",
                    vec![f.e()],
                    anys.clone(),
                ))],
            ),
            // A foreign object is called by the embedder, through the
            // caller with the object in front of the arguments.
            when(
                zyntax_builtins::foreign::is_foreign(f.e()),
                vec![
                    expr(mcall(args.e(), "insert_at", vec![int(0), f.e()], unit())),
                    ret(call(
                        "zb_unbox_list_raw_any",
                        vec![call("zl_foreign_caller", vec![], any())],
                        anys.clone(),
                    )),
                ],
            ),
            // A value of another type is called through its type's
            // `__call` (`debug.setmetatable`).
            when(
                not(is_table(f.e())),
                vec![
                    y.decl(call("zl_meta_of", vec![f.e(), text("__call")], any())),
                    when(
                        is_func(y.e()),
                        vec![
                            expr(mcall(args.e(), "insert_at", vec![int(0), f.e()], unit())),
                            ret(call("zb_unbox_list_raw_any", vec![y.e()], anys.clone())),
                        ],
                    ),
                ],
            ),
            // A table's `__call` handler, which may be a table with
            // one of its own: each puts its object in front, as many
            // times as the C stack allows.
            h.decl(f.e()),
            i.decl(int(0)),
            while_(
                and(is_table(h.e()), lt(i.e(), int(CCALLS_LIMIT))),
                vec![
                    expr(mcall(args.e(), "insert_at", vec![int(0), h.e()], unit())),
                    h.set(call("zl_meta_of", vec![h.e(), text("__call")], any())),
                    when(
                        is_func(h.e()),
                        vec![ret(call(
                            "zb_unbox_list_raw_any",
                            vec![h.e()],
                            anys.clone(),
                        ))],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            // A handler that is not a function is what the error
            // names, as the reference tries to call it; a table with
            // no handler is itself, put in front of the arguments.
            when(
                and(is_nil(h.e()), gt(i.e(), int(0))),
                vec![h.set(at(args.e(), int(0)))],
            ),
            not_callable(&h),
            ret(list(vec![], anys.clone())),
        ],
    ));
    // The call itself, from a record and a list of arguments adjusted
    // to what the code takes.
    let variadic_call = |rec: &Local, args: Expr| {
        let packed_fp = local("packed_fp", code_type(t.list_type, 1));
        vec![
            packed_fp.decl(call(
                &fp_name(1),
                vec![at(rec.e(), int(0))],
                code_type(t.list_type, 1),
            )),
            ret(call(
                "packed_fp",
                vec![rec.e(), call("zb_box_tuple", vec![args], any())],
                any(),
            )),
        ]
    };
    d.push(define("zl_apply_packed", &[&f, &args], any(), {
        let mut st = vec![
            rec.decl(call("zl_callee", vec![f.e(), args.e()], anys.clone())),
            // Nothing callable: the error is raised, the call is nil.
            when(eq(len(rec.e()), int(0)), vec![ret(nil())]),
            arity.decl(call("zb_box_get_i64", vec![at(rec.e(), int(1))], i64())),
            when(
                eq(arity.e(), int(VARIADIC_ARITY)),
                variadic_call(&rec, args.e()),
            ),
            most.decl(bitand(arity.e(), int(0xFFFF))),
            // Fewer arguments than parameters: nil for the rest. More:
            // the extra ones go.
            while_(lt(len(args.e()), most.e()), vec![push(args.e(), nil())]),
        ];
        for m in 0..=MAX_CALL_ARITY {
            let code_ty = code_type(t.list_type, m);
            let fp = local("fp", code_ty.clone());
            let mut call_args = vec![rec.e()];
            call_args.extend((0..m).map(|k| at(args.e(), int(k as i64))));
            st.push(when(
                eq(most.e(), int(m as i64)),
                vec![
                    fp.decl(call(&fp_name(m), vec![at(rec.e(), int(0))], code_ty)),
                    ret(call("fp", call_args, any())),
                ],
            ));
        }
        st.push(lua_error(text(
            "a function of more than 8 parameters cannot be called through a value",
        )));
        st.push(ret(nil()));
        st
    }));
    // One entry per argument count: the common case, a record of that
    // arity, is a tag check and the call.
    let params: Vec<Local> = (0..MAX_CALL_ARITY)
        .map(|i| {
            let name: &'static str = Box::leak(format!("a{i}").into_boxed_str());
            kept(name, any())
        })
        .collect();
    for n in 0..=MAX_CALL_ARITY {
        let code_ty = code_type(t.list_type, n);
        let fp = local("fp", code_ty.clone());
        let mut sig: Vec<&Local> = vec![&f];
        sig.extend(params[..n].iter());
        let mut direct = vec![rec.e()];
        direct.extend(params[..n].iter().map(|p| p.e()));
        d.push(define(
            &format!("zl_apply_{n}"),
            &sig,
            any(),
            vec![
                when(
                    is_func(f.e()),
                    vec![
                        rec.decl(call("zb_unbox_list_raw_any", vec![f.e()], anys.clone())),
                        arity.decl(call("zb_box_get_i64", vec![at(rec.e(), int(1))], i64())),
                        when(
                            eq(arity.e(), int(n as i64)),
                            vec![
                                fp.decl(call(&fp_name(n), vec![at(rec.e(), int(0))], code_ty)),
                                ret(call("fp", direct, any())),
                            ],
                        ),
                    ],
                ),
                ret(call(
                    "zl_apply_packed",
                    vec![
                        f.e(),
                        list(params[..n].iter().map(|p| p.e()).collect(), anys.clone()),
                    ],
                    any(),
                )),
            ],
        ));
    }
    // The same from the library: a metamethod, an iterator, a
    // comparator, a handler. These nest on the reference's C stack,
    // so they are counted and refused past its limit; a call the
    // program makes itself is not, as its own recursion is bounded by
    // the depth.
    let ccalls = || read_global(CCALLS, i64());
    let counted = |name: &str, apply: &str, sig: &[&Local], args: Vec<Expr>| {
        define(
            name,
            sig,
            any(),
            vec![
                when(
                    ge(ccalls(), int(CCALLS_LIMIT)),
                    vec![lua_error(text("C stack overflow")), ret(nil())],
                ),
                set_global(CCALLS, add(ccalls(), int(1))),
                y.decl(call(apply, args, any())),
                set_global(CCALLS, sub(ccalls(), int(1))),
                ret(y.e()),
            ],
        )
    };
    d.push(counted(
        "zl_call_packed",
        "zl_apply_packed",
        &[&f, &args],
        vec![f.e(), args.e()],
    ));
    for n in 0..=MAX_CALL_ARITY {
        let mut sig: Vec<&Local> = vec![&f];
        sig.extend(params[..n].iter());
        let mut args = vec![f.e()];
        args.extend(params[..n].iter().map(|p| p.e()));
        d.push(counted(
            &format!("zl_call_{n}"),
            &format!("zl_apply_{n}"),
            &sig,
            args,
        ));
    }
    // A record for a library function's code, so a program can pass
    // `print` around: no cells, the arity given.
    let code = local("code", usize());
    d.push(define(
        "zl_func_of",
        &[&code, &arity],
        any(),
        vec![ret(call(
            "zb_func_new",
            vec![code.e(), arity.e(), list(vec![], anys.clone())],
            any(),
        ))],
    ));
    d
}
