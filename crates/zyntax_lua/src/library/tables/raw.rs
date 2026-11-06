//! Raw declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        arr,
        cat,
        f,
        h,
        handler,
        i,
        k,
        n,
        s,
        table,
        tb,
        v,
        x,
        ..
    } = context;
    let arr_of = |tb: Expr| crate::library::arr_of(tb, t);
    let hash_of = |tb: Expr| crate::library::hash_of(tb, t);
    // `arr` is the array part boxed.
    let struct_lit = |arr: Expr, high: Expr| {
        use zyntax_typed_ast::typed_ast::{TypedExpression, TypedFieldInit, TypedStructLiteral};
        node(
            TypedExpression::Struct(TypedStructLiteral {
                name: intern(TABLE_TYPE),
                fields: vec![
                    TypedFieldInit {
                        name: intern("arr"),
                        value: Box::new(arr),
                    },
                    TypedFieldInit {
                        name: intern("hash"),
                        value: Box::new(nil()),
                    },
                    TypedFieldInit {
                        name: intern("meta"),
                        value: Box::new(null(table.clone())),
                    },
                    TypedFieldInit {
                        name: intern("high"),
                        value: Box::new(high),
                    },
                    TypedFieldInit {
                        name: intern("shape"),
                        value: Box::new(int(0)),
                    },
                    TypedFieldInit {
                        name: intern("present"),
                        value: Box::new(int(0)),
                    },
                ],
            }),
            table.clone(),
        )
    };
    let shaped = |tb: Expr| crate::library::is_shaped(tb);
    let slot_index = |tb: Expr, name: Expr| call("zl_shape_index", vec![tb, name], i64());
    let slot_load = |tb: Expr, i: Expr| call("zl_shape_load", vec![tb, i], any());
    // A store into a slot of a value not of its kind is a hole in the
    // static rules that keep the two apart; it is reported, not made.
    let slot_store = |tb: Expr, i: Expr, v: Expr| {
        when(
            ne(call("zl_shape_store", vec![tb, i, v], i64()), int(0)),
            vec![lua_error(text(
                "internal: a value of another kind stored into a shaped field",
            ))],
        )
    };

    // ─── construction ───────────────────────────────────────────
    // The empty array part every table starts with, shared: made on
    // first use and never written, since a write gives the table its
    // own first.
    d.push(define(
        "zl_arr_shared",
        &[],
        any(),
        vec![
            when(
                is_nil(read_global(ARR_EMPTY, any())),
                vec![set_global(
                    ARR_EMPTY,
                    call(
                        "zb_list_box_any",
                        vec![list(Vec::new(), anys.clone())],
                        any(),
                    ),
                )],
            ),
            ret(read_global(ARR_EMPTY, any())),
        ],
    ));
    // The array part of `t` for writing: its own, made when it still
    // holds the shared empty one.
    d.push(define(
        "zl_arr_own",
        &[tb],
        anys.clone(),
        vec![
            when(
                eq(arr_field(tb.e()), read_global(ARR_EMPTY, any())),
                vec![set_field(
                    tb.e(),
                    "arr",
                    call(
                        "zb_list_box_any",
                        vec![list(Vec::new(), anys.clone())],
                        any(),
                    ),
                )],
            ),
            ret(arr_of(tb.e())),
        ],
    ));
    d.push(define(
        "zl_table_new",
        &[],
        table.clone(),
        vec![ret(struct_lit(
            call("zl_arr_shared", vec![], any()),
            int(0),
        ))],
    ));
    // A table over the positional values of a constructor, which are
    // its array part as they are: a trailing nil is dropped so the
    // length stays a border.
    let arr_kept = kept("arr", anys.clone());
    d.push(define("zl_table_with_arr", &[&arr_kept], table.clone(), {
        vec![
            n.decl(len(arr.e())),
            ret(struct_lit(call("zl_arr_box", vec![arr.e()], any()), n.e())),
        ]
    }));
    // The array part of a shaped table, boxed, trimmed like a plain
    // table's; the program lays the rest of the table out itself. An
    // empty one is the shared one.
    d.push(define("zl_arr_box", &[&arr_kept], any(), {
        vec![
            while_(
                and(
                    gt(len(arr.e()), int(0)),
                    is_nil(at(arr.e(), sub(len(arr.e()), int(1)))),
                ),
                vec![expr(mcall(arr.e(), "pop_last", vec![], any()))],
            ),
            when(
                eq(len(arr.e()), int(0)),
                vec![ret(call("zl_arr_shared", vec![], any()))],
            ),
            ret(call("zb_list_box_any", vec![arr.e()], any())),
        ]
    }));
    // The hash part, made on first use.
    d.push(define(
        "zl_hash_ensure",
        &[tb],
        t.dict(),
        vec![
            when(
                is_nil(hash_field(tb.e())),
                vec![set_field(
                    tb.e(),
                    "hash",
                    call(
                        "zb_dict_box",
                        vec![call("zb_dict_new", vec![], t.dict())],
                        any(),
                    ),
                )],
            ),
            ret(hash_of(tb.e())),
        ],
    ));

    // ─── raw reads ──────────────────────────────────────────────
    d.push(define(
        "zl_rawgeti",
        &[tb, i],
        any(),
        vec![
            arr.decl(arr_of(tb.e())),
            when(
                and(ge(i.e(), int(1)), le(i.e(), len(arr.e()))),
                vec![ret(at(arr.e(), sub(i.e(), int(1))))],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(nil())]),
            ret(call(
                "zb_dict_get_default",
                vec![hash_of(tb.e()), box_i64(i.e()), nil()],
                any(),
            )),
        ],
    ));
    let slot = local("slot", i64());
    d.push(define(
        "zl_rawget_str",
        &[tb, s],
        any(),
        vec![
            // A shaped table's constant-key fields are its slots.
            when(
                shaped(tb.e()),
                vec![
                    slot.decl(slot_index(tb.e(), s.e())),
                    when(ge(slot.e(), int(0)), vec![ret(slot_load(tb.e(), slot.e()))]),
                ],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(nil())]),
            ret(call(
                "zb_dict_get_default_str",
                vec![hash_of(tb.e()), s.e(), nil()],
                any(),
            )),
        ],
    ));
    // A key boxed once for a name the program spells: the box is
    // shared by every use of that name, so a lookup meets it by
    // identity before comparing text, and a store keeps it.
    d.push(define(
        "zl_rawget_key",
        &[tb, k],
        any(),
        vec![
            when(
                shaped(tb.e()),
                vec![
                    slot.decl(slot_index(tb.e(), get_str(k.e()))),
                    when(ge(slot.e(), int(0)), vec![ret(slot_load(tb.e(), slot.e()))]),
                ],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(nil())]),
            ret(call(
                "zb_dict_get_default",
                vec![hash_of(tb.e()), k.e(), nil()],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_rawset_key",
        &[tb, k, v],
        unit(),
        vec![
            when(
                shaped(tb.e()),
                vec![
                    slot.decl(slot_index(tb.e(), get_str(k.e()))),
                    when(
                        ge(slot.e(), int(0)),
                        vec![slot_store(tb.e(), slot.e(), v.e()), ret_void()],
                    ),
                ],
            ),
            when(
                and(is_nil(v.e()), is_nil(hash_field(tb.e()))),
                vec![ret_void()],
            ),
            when(
                ne(read_global(crate::library::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), k.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![call("zl_hash_ensure", vec![tb.e()], t.dict()), k.e(), v.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // A float key with an integral value is that integer; NaN and nil
    // are never present.
    d.push(define(
        "zl_rawget",
        &[tb, k],
        any(),
        vec![
            when(is_nil(k.e()), vec![ret(nil())]),
            cat.decl(category(k.e())),
            when(
                is_int_cat(cat),
                vec![ret(call("zl_rawgeti", vec![tb.e(), get_i64(k.e())], any()))],
            ),
            when(
                is_cat(cat, STR),
                vec![ret(call(
                    "zl_rawget_str",
                    vec![tb.e(), get_str(k.e())],
                    any(),
                ))],
            ),
            when(
                is_cat(cat, FLOAT),
                vec![
                    f.decl(get_f64(k.e())),
                    when(
                        call("zl_float_is_int", vec![f.e()], boolean()),
                        vec![ret(call(
                            "zl_rawgeti",
                            vec![tb.e(), cast(f.e(), i64())],
                            any(),
                        ))],
                    ),
                ],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(nil())]),
            ret(call(
                "zb_dict_get_default",
                vec![hash_of(tb.e()), k.e(), nil()],
                any(),
            )),
        ],
    ));

    // ─── raw writes ─────────────────────────────────────────────
    // Keys the hash part holds that now belong to the array part, after
    // an append made `len + 1` the next key.
    d.push(define(
        "zl_migrate",
        &[tb],
        unit(),
        vec![
            when(is_nil(hash_field(tb.e())), vec![ret_void()]),
            h.decl(hash_of(tb.e())),
            arr.decl(call("zl_arr_own", vec![tb.e()], anys.clone())),
            when(
                eq(call("zb_dict_len", vec![h.e()], i64()), int(0)),
                vec![ret_void()],
            ),
            x.decl(call(
                "zb_dict_get_default",
                vec![h.e(), box_i64(add(len(arr.e()), int(1))), nil()],
                any(),
            )),
            while_(
                not(is_nil(x.e())),
                vec![
                    push(arr.e(), x.e()),
                    expr(call(
                        "zb_dict_set",
                        vec![h.e(), box_i64(len(arr.e())), nil()],
                        unit(),
                    )),
                    x.set(call(
                        "zb_dict_get_default",
                        vec![h.e(), box_i64(add(len(arr.e()), int(1))), nil()],
                        any(),
                    )),
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_rawseti",
        &[tb, i, v],
        unit(),
        vec![
            when(
                eq(arr_field(tb.e()), read_global(ARR_EMPTY, any())),
                vec![expr(call("zl_arr_own", vec![tb.e()], anys.clone()))],
            ),
            arr.decl(arr_of(tb.e())),
            n.decl(len(arr.e())),
            when(
                and(ge(i.e(), int(1)), le(i.e(), n.e())),
                vec![
                    if_(
                        and(is_nil(v.e()), eq(i.e(), n.e())),
                        vec![
                            when(
                                gt(n.e(), crate::library::high_of(tb.e())),
                                vec![set_field(tb.e(), "high", n.e())],
                            ),
                            expr(mcall(arr.e(), "pop_last", vec![], any())),
                            while_(
                                and(
                                    gt(len(arr.e()), int(0)),
                                    is_nil(at(arr.e(), sub(len(arr.e()), int(1)))),
                                ),
                                vec![expr(mcall(arr.e(), "pop_last", vec![], any()))],
                            ),
                        ],
                        vec![set_idx(arr.e(), sub(i.e(), int(1)), v.e())],
                    ),
                    ret_void(),
                ],
            ),
            when(
                and(eq(i.e(), add(n.e(), int(1))), not(is_nil(v.e()))),
                vec![
                    push(arr.e(), v.e()),
                    when(
                        not(is_nil(hash_field(tb.e()))),
                        vec![expr(call("zl_migrate", vec![tb.e()], unit()))],
                    ),
                    ret_void(),
                ],
            ),
            // Absent, and told to stay absent.
            when(
                and(is_nil(v.e()), is_nil(hash_field(tb.e()))),
                vec![ret_void()],
            ),
            when(
                ne(read_global(crate::library::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), box_i64(i.e()), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![
                    call("zl_hash_ensure", vec![tb.e()], t.dict()),
                    box_i64(i.e()),
                    v.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_rawset_str",
        &[tb, s, v],
        unit(),
        vec![
            when(
                shaped(tb.e()),
                vec![
                    slot.decl(slot_index(tb.e(), s.e())),
                    when(
                        ge(slot.e(), int(0)),
                        vec![slot_store(tb.e(), slot.e(), v.e()), ret_void()],
                    ),
                ],
            ),
            when(
                and(is_nil(v.e()), is_nil(hash_field(tb.e()))),
                vec![ret_void()],
            ),
            when(
                ne(read_global(crate::library::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert_str",
                    vec![tb.e(), s.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set_str",
                vec![call("zl_hash_ensure", vec![tb.e()], t.dict()), s.e(), v.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_rawset",
        &[tb, k, v],
        unit(),
        vec![
            when(is_nil(k.e()), vec![lua_error(text("table index is nil"))]),
            cat.decl(category(k.e())),
            when(
                is_int_cat(cat),
                vec![
                    expr(call(
                        "zl_rawseti",
                        vec![tb.e(), get_i64(k.e()), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                is_cat(cat, STR),
                vec![
                    expr(call(
                        "zl_rawset_str",
                        vec![tb.e(), get_str(k.e()), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                is_cat(cat, FLOAT),
                vec![
                    f.decl(get_f64(k.e())),
                    when(
                        ne(f.e(), f.e()),
                        vec![lua_error(text("table index is NaN"))],
                    ),
                    when(
                        call("zl_float_is_int", vec![f.e()], boolean()),
                        vec![
                            expr(call(
                                "zl_rawseti",
                                vec![tb.e(), cast(f.e(), i64()), v.e()],
                                unit(),
                            )),
                            ret_void(),
                        ],
                    ),
                ],
            ),
            when(
                and(is_nil(v.e()), is_nil(hash_field(tb.e()))),
                vec![ret_void()],
            ),
            when(
                ne(read_global(crate::library::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), k.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![call("zl_hash_ensure", vec![tb.e()], t.dict()), k.e(), v.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));

    // `#t`: the array part's length, a border by construction.
    d.push(define(
        "zl_len",
        &[tb],
        i64(),
        vec![ret(len(arr_of(tb.e())))],
    ));
    // `#t` with `__len`, as an integer: what the table library
    // measures a table by, which a value the length cannot be read as
    // one refuses.
    d.push(define(
        "zl_table_len",
        &[tb],
        i64(),
        vec![
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret(len(arr_of(tb.e())))],
            ),
            handler.decl(call("zl_meta", vec![tb.e(), text("__len")], any())),
            when(is_nil(handler.e()), vec![ret(len(arr_of(tb.e())))]),
            metamethod_call_check(handler.e(), text("len")),
            x.decl(call(
                "zl_math_tointeger",
                vec![call(
                    "zl_first",
                    vec![call(
                        "zl_call_2",
                        vec![handler.e(), box_table(tb.e()), box_table(tb.e())],
                        any(),
                    )],
                    any(),
                )],
                any(),
            )),
            when(
                is_nil(x.e()),
                vec![lua_error(text("object length is not an integer"))],
            ),
            ret(get_i64(x.e())),
        ],
    ));
}
