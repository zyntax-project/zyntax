//! Table declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        i,
        j,
        k,
        n,
        out,
        s,
        sep,
        table,
        tb,
        ..
    } = context;
    // `table.unpack(t, i, j)`: the values `t[i]..t[j]`, `j` the
    // length when not given; any integer may be given.
    let ok = local("ok", boolean());
    let last = kept("last", any());
    let is_int_box = |x: Expr| {
        let c = category(x);
        or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
    };

    // ─── table ──────────────────────────────────────────────────
    // `table.insert(t, v)` appends; `table.insert(t, pos, v)` shifts
    // what follows up.
    let arr = borrowed("arr", anys.clone());
    let v = kept("v", any());
    d.push(define(
        "zl_table_insert",
        &[tb, args],
        unit(),
        vec![
            n.decl(call("zl_table_len", vec![tb.e()], i64())),
            when(
                eq(len(args.e()), int(1)),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![tb.e(), add(n.e(), int(1)), at(args.e(), int(0))],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                ne(len(args.e()), int(2)),
                vec![lua_error(text("wrong number of arguments to 'insert'"))],
            ),
            i.decl(call(
                "zl_arg_int",
                vec![at(args.e(), int(0)), bad_arg(2, "insert")],
                i64(),
            )),
            when(
                or(lt(i.e(), int(1)), gt(i.e(), add(n.e(), int(1)))),
                vec![lua_error(text(
                    "bad argument #2 to 'insert' (position out of bounds)",
                ))],
            ),
            v.decl(at(args.e(), int(1))),
            when(
                eq(i.e(), add(n.e(), int(1))),
                vec![
                    expr(call("zl_table_seti", vec![tb.e(), i.e(), v.e()], unit())),
                    ret_void(),
                ],
            ),
            // Within the array part of a table without a metatable:
            // shift in place.
            arr.decl(call("zl_arr_own", vec![tb.e()], t.anys())),
            when(
                and(
                    eq(len(arr.e()), n.e()),
                    eq(crate::library::meta_of(tb.e(), t), null(table.clone())),
                ),
                vec![
                    expr(mcall(
                        arr.e(),
                        "insert_at",
                        vec![sub(i.e(), int(1)), v.e()],
                        unit(),
                    )),
                    expr(call("zl_migrate", vec![tb.e()], unit())),
                    ret_void(),
                ],
            ),
            k.decl(n.e()),
            while_(
                ge(k.e(), i.e()),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![
                            tb.e(),
                            add(k.e(), int(1)),
                            call("zl_table_geti", vec![tb.e(), k.e()], any()),
                        ],
                        unit(),
                    )),
                    k.set(sub(k.e(), int(1))),
                ],
            ),
            expr(call("zl_table_seti", vec![tb.e(), i.e(), v.e()], unit())),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_remove",
        &[tb, i],
        any(),
        vec![
            n.decl(call("zl_table_len", vec![tb.e()], i64())),
            when(eq(i.e(), int(i64::MIN)), vec![i.set(n.e())]),
            // An empty table: the entry at 0 or 1, taken out as well.
            when(
                and(eq(n.e(), int(0)), or(eq(i.e(), int(0)), eq(i.e(), n.e()))),
                vec![
                    v.decl(call("zl_table_geti", vec![tb.e(), i.e()], any())),
                    expr(call("zl_table_seti", vec![tb.e(), i.e(), nil()], unit())),
                    ret(v.e()),
                ],
            ),
            when(
                or(lt(i.e(), int(1)), gt(i.e(), add(n.e(), int(1)))),
                vec![lua_error(text(
                    "bad argument #2 to 'remove' (position out of bounds)",
                ))],
            ),
            v.decl(call("zl_table_geti", vec![tb.e(), i.e()], any())),
            arr.decl(call("zl_arr_own", vec![tb.e()], t.anys())),
            if_(
                and(
                    and(eq(len(arr.e()), n.e()), le(i.e(), n.e())),
                    eq(crate::library::meta_of(tb.e(), t), null(table.clone())),
                ),
                vec![
                    when(
                        gt(n.e(), crate::library::high_of(tb.e())),
                        vec![set_field(tb.e(), "high", n.e())],
                    ),
                    expr(mcall(arr.e(), "remove_at", vec![sub(i.e(), int(1))], any())),
                    while_(
                        and(
                            gt(len(arr.e()), int(0)),
                            is_nil(at(arr.e(), sub(len(arr.e()), int(1)))),
                        ),
                        vec![expr(mcall(arr.e(), "pop_last", vec![], any()))],
                    ),
                ],
                vec![
                    k.decl(i.e()),
                    while_(
                        lt(k.e(), n.e()),
                        vec![
                            expr(call(
                                "zl_table_seti",
                                vec![
                                    tb.e(),
                                    k.e(),
                                    call("zl_table_geti", vec![tb.e(), add(k.e(), int(1))], any()),
                                ],
                                unit(),
                            )),
                            k.add_assign(int(1)),
                        ],
                    ),
                    when(
                        le(i.e(), n.e()),
                        vec![expr(call(
                            "zl_table_seti",
                            vec![tb.e(), n.e(), nil()],
                            unit(),
                        ))],
                    ),
                ],
            ),
            ret(v.e()),
        ],
    ));
    // `table.concat(t, sep, i, j)`: the last index defaults to the
    // length; the loop stops on reaching it rather than passing it, so
    // an index at either end of the integers is fine. The pieces are
    // gathered and joined in one allocation of the result's length.
    let parts = borrowed("parts", zyntax_builtins::list_of(t.list_type, string()));
    d.push(define(
        "zl_table_concat",
        &[tb, sep, i, &last],
        string(),
        vec![
            j.decl(if_expr(
                is_nil(last.e()),
                call("zl_table_len", vec![tb.e()], i64()),
                call("zl_arg_int", vec![last.e(), bad_arg(4, "concat")], i64()),
            )),
            when(gt(i.e(), j.e()), vec![ret(text(""))]),
            parts.decl(list(Vec::new(), parts.ty.clone())),
            k.decl(i.e()),
            ok.decl(bool(true)),
            while_(
                ok.e(),
                vec![
                    v.decl(call("zl_table_geti", vec![tb.e(), k.e()], any())),
                    when(
                        or(
                            is_nil(v.e()),
                            not(or(
                                eq(category(v.e()), int(STR)),
                                or(is_int_box(v.e()), eq(category(v.e()), int(FLOAT))),
                            )),
                        ),
                        vec![lua_error(concat(vec![
                            text("invalid value ("),
                            call("zl_type", vec![v.e()], string()),
                            text(") at index "),
                            call("zb_str_of_int", vec![k.e()], string()),
                            text(" in table for 'concat'"),
                        ]))],
                    ),
                    push(parts.e(), call("zl_concat_text", vec![v.e()], string())),
                    if_(
                        eq(k.e(), j.e()),
                        vec![ok.set(bool(false))],
                        vec![k.add_assign(int(1))],
                    ),
                ],
            ),
            ret(call("zb_str_join", vec![sep.e(), parts.e()], string())),
        ],
    ));
    d.push(define(
        "zl_table_pack",
        &[args],
        table.clone(),
        vec![
            tb.decl(call(
                "zl_table_with_arr",
                vec![call("zb_list_copy_any", vec![args.e()], anys.clone())],
                table.clone(),
            )),
            expr(call(
                "zl_rawset_str",
                vec![tb.e(), text("n"), box_i64(len(args.e()))],
                unit(),
            )),
            ret(tb.e()),
        ],
    ));
    // `table.move(a1, f, e, t, a2)`: `a1[f..e]` copied to `a2[t..]`,
    // `a2` being `a1` when not given; the destination is returned.
    // Overlapping ranges in one table are copied from the end when
    // the move is forward, so nothing is overwritten before it is
    // read.
    let from = local("from", i64());
    let to = local("to", i64());
    let at_i = local("at_i", i64());
    let dest = kept("dest", any());
    let target = kept("target", table.clone());
    d.push(define(
        "zl_table_move",
        &[tb, &from, &to, &at_i, &dest],
        any(),
        vec![
            target.decl(if_expr(
                is_nil(dest.e()),
                tb.e(),
                call(
                    "zl_as_table",
                    vec![dest.e(), bad_arg(5, "move")],
                    table.clone(),
                ),
            )),
            when(
                ge(to.e(), from.e()),
                vec![
                    when(
                        and(
                            le(from.e(), int(0)),
                            ge(to.e(), add(int(i64::MAX), from.e())),
                        ),
                        vec![lua_error(text(
                            "bad argument #3 to 'move' (too many elements to move)",
                        ))],
                    ),
                    n.decl(add(sub(to.e(), from.e()), int(1))),
                    when(
                        gt(at_i.e(), add(sub(int(i64::MAX), n.e()), int(1))),
                        vec![lua_error(text(
                            "bad argument #4 to 'move' (destination wrap around)",
                        ))],
                    ),
                    // A metamethod's error ends the move.
                    if_(
                        or(
                            or(gt(at_i.e(), to.e()), le(at_i.e(), from.e())),
                            ne(target.e(), tb.e()),
                        ),
                        vec![
                            i.decl(int(0)),
                            while_(
                                and(lt(i.e(), n.e()), is_nil(pending())),
                                vec![
                                    expr(call(
                                        "zl_table_seti",
                                        vec![
                                            target.e(),
                                            add(at_i.e(), i.e()),
                                            call(
                                                "zl_table_geti",
                                                vec![tb.e(), add(from.e(), i.e())],
                                                any(),
                                            ),
                                        ],
                                        unit(),
                                    )),
                                    i.add_assign(int(1)),
                                ],
                            ),
                        ],
                        vec![
                            i.decl(sub(n.e(), int(1))),
                            while_(
                                and(ge(i.e(), int(0)), is_nil(pending())),
                                vec![
                                    expr(call(
                                        "zl_table_seti",
                                        vec![
                                            target.e(),
                                            add(at_i.e(), i.e()),
                                            call(
                                                "zl_table_geti",
                                                vec![tb.e(), add(from.e(), i.e())],
                                                any(),
                                            ),
                                        ],
                                        unit(),
                                    )),
                                    i.set(sub(i.e(), int(1))),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            ret(box_table(target.e())),
        ],
    ));
    // `table.sort(t, comp)`: a quicksort over the array part, ordering
    // by `comp` or `<`.
    let comp = kept("comp", any());
    let lo_i = local("lo", i64());
    let hi_i = local("hi", i64());
    let pivot = kept("pivot", any());
    let tmp = kept("tmp", any());
    let less = |a: Expr, b: Expr| {
        if_expr(
            is_nil(comp.e()),
            call("zl_lt", vec![a.clone(), b.clone()], boolean()),
            call(
                "zl_truthy",
                vec![call(
                    "zl_first",
                    vec![call("zl_call_2", vec![comp.e(), a, b], any())],
                    any(),
                )],
                boolean(),
            ),
        )
    };
    // `line` is the sort's own, for the error it raises after the
    // comparator has moved the line elsewhere.
    let line = local("line", i64());
    d.push(define(
        "zl_sort_range",
        &[&arr, &lo_i, &hi_i, &comp, &line],
        unit(),
        vec![
            when(ge(lo_i.e(), hi_i.e()), vec![ret_void()]),
            pivot.decl(at(arr.e(), div(add(lo_i.e(), hi_i.e()), int(2)))),
            i.decl(lo_i.e()),
            j.decl(hi_i.e()),
            while_(
                le(i.e(), j.e()),
                vec![
                    // An order that never settles walks off the range:
                    // the comparator is not one.
                    while_(
                        and(le(i.e(), hi_i.e()), less(at(arr.e(), i.e()), pivot.e())),
                        vec![i.add_assign(int(1))],
                    ),
                    while_(
                        and(ge(j.e(), lo_i.e()), less(pivot.e(), at(arr.e(), j.e()))),
                        vec![j.set(sub(j.e(), int(1)))],
                    ),
                    when(
                        or(gt(i.e(), hi_i.e()), lt(j.e(), lo_i.e())),
                        vec![
                            set_global(LINE, line.e()),
                            lua_error(text("invalid order function for sorting")),
                        ],
                    ),
                    when(
                        le(i.e(), j.e()),
                        vec![
                            tmp.decl(at(arr.e(), i.e())),
                            set_idx(arr.e(), i.e(), at(arr.e(), j.e())),
                            set_idx(arr.e(), j.e(), tmp.e()),
                            i.add_assign(int(1)),
                            j.set(sub(j.e(), int(1))),
                        ],
                    ),
                ],
            ),
            when(not(is_nil(pending())), vec![ret_void()]),
            expr(call(
                "zl_sort_range",
                vec![arr.e(), lo_i.e(), j.e(), comp.e(), line.e()],
                unit(),
            )),
            expr(call(
                "zl_sort_range",
                vec![arr.e(), i.e(), hi_i.e(), comp.e(), line.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_sort",
        &[tb, &comp],
        unit(),
        vec![
            line.decl(read_global(LINE, i64())),
            when(
                eq(crate::library::meta_of(tb.e(), t), null(table.clone())),
                vec![
                    arr.decl(call("zl_arr_own", vec![tb.e()], t.anys())),
                    expr(call(
                        "zl_sort_range",
                        vec![
                            arr.e(),
                            int(0),
                            sub(len(arr.e()), int(1)),
                            comp.e(),
                            line.e(),
                        ],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            // With a metatable, `t[1..n]` are read and written back
            // through it, sorted in between.
            n.decl(call("zl_table_len", vec![tb.e()], i64())),
            when(
                ge(n.e(), int(i32::MAX as i64)),
                vec![lua_error(text("bad argument #1 to 'sort' (array too big)"))],
            ),
            out.decl(list(vec![], anys.clone())),
            k.decl(int(1)),
            while_(
                le(k.e(), n.e()),
                vec![
                    push(out.e(), call("zl_table_geti", vec![tb.e(), k.e()], any())),
                    k.add_assign(int(1)),
                ],
            ),
            expr(call(
                "zl_sort_range",
                vec![out.e(), int(0), sub(n.e(), int(1)), comp.e(), line.e()],
                unit(),
            )),
            k.set(int(1)),
            while_(
                le(k.e(), n.e()),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![tb.e(), k.e(), at(out.e(), sub(k.e(), int(1)))],
                        unit(),
                    )),
                    k.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
    ));
    // `package.searchpath(name, path [, sep [, rep]])`: the file found,
    // or nil and the list of files tried.
    let sep = kept("sep", string());
    let repeated = kept("repeated", string());
    d.push(define(
        "zl_string_rep_of",
        &[s, n, &sep],
        string(),
        vec![
            repeated.decl(call("zl_string_rep", vec![s.e(), n.e(), sep.e()], string())),
            when(
                eq(repeated.e(), null(string())),
                vec![lua_error(text("resulting string too large"))],
            ),
            ret(repeated.e()),
        ],
    ));
}
