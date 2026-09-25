//! Tables: the array part, the hash part, the border `#` reports,
//! metatables and the `__index` / `__newindex` events.
//!
//! Keys `1..=n` live in the array part; every other key in the hash
//! part, a dict of the shared library. A float key with an integral
//! value is the integer key. Storing nil under a hash key keeps the
//! key with a nil value, so a traversal that clears entries as it goes
//! sees every one; storing nil at the end of the array part pops it and
//! whatever nils it uncovers, so the array's length stays a border.

use super::*;

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let tb = kept("t", table.clone());
    let o = kept("o", any());
    let k = kept("k", any());
    let v = kept("v", any());
    let s = kept("s", string());
    let i = local("i", i64());
    let n = local("n", i64());
    let arr = borrowed("arr", anys.clone());
    let h = borrowed("h", anys.clone());
    let x = kept("x", any());
    let f = local("f", f64());
    let cat = local("cat", i64());
    let mt = kept("mt", any());
    let handler = kept("handler", any());
    let mut d = Vec::new();

    let arr_of = |tb: Expr| super::arr_of(tb, t);
    let hash_of = |tb: Expr| super::hash_of(tb, t);
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
    let shaped = |tb: Expr| super::is_shaped(tb);
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
    let slot_count = |tb: Expr| call("zl_shape_count", vec![tb], i64());
    let slot_key = |tb: Expr, i: Expr| call("zl_shape_key", vec![tb, i], string());

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
        &[&tb],
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
        &[&tb],
        anys.clone(),
        vec![
            when(
                is_nil(hash_field(tb.e())),
                vec![set_field(
                    tb.e(),
                    "hash",
                    call(
                        "zb_list_box_any",
                        vec![call("zb_dict_new", vec![], anys.clone())],
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
        &[&tb, &i],
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
        &[&tb, &s],
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
        &[&tb, &k],
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
        &[&tb, &k, &v],
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
                ne(read_global(super::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), k.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![
                    call("zl_hash_ensure", vec![tb.e()], anys.clone()),
                    k.e(),
                    v.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // A float key with an integral value is that integer; NaN and nil
    // are never present.
    d.push(define(
        "zl_rawget",
        &[&tb, &k],
        any(),
        vec![
            when(is_nil(k.e()), vec![ret(nil())]),
            cat.decl(category(k.e())),
            when(
                is_int_cat(&cat),
                vec![ret(call("zl_rawgeti", vec![tb.e(), get_i64(k.e())], any()))],
            ),
            when(
                is_cat(&cat, STR),
                vec![ret(call(
                    "zl_rawget_str",
                    vec![tb.e(), get_str(k.e())],
                    any(),
                ))],
            ),
            when(
                is_cat(&cat, FLOAT),
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
        &[&tb],
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
        &[&tb, &i, &v],
        unit(),
        vec![
            arr.decl(call("zl_arr_own", vec![tb.e()], anys.clone())),
            n.decl(len(arr.e())),
            when(
                and(ge(i.e(), int(1)), le(i.e(), n.e())),
                vec![
                    if_(
                        and(is_nil(v.e()), eq(i.e(), n.e())),
                        vec![
                            when(
                                gt(n.e(), super::high_of(tb.e())),
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
                    expr(call("zl_migrate", vec![tb.e()], unit())),
                    ret_void(),
                ],
            ),
            // Absent, and told to stay absent.
            when(
                and(is_nil(v.e()), is_nil(hash_field(tb.e()))),
                vec![ret_void()],
            ),
            when(
                ne(read_global(super::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), box_i64(i.e()), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![
                    call("zl_hash_ensure", vec![tb.e()], anys.clone()),
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
        &[&tb, &s, &v],
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
                ne(read_global(super::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert_str",
                    vec![tb.e(), s.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set_str",
                vec![
                    call("zl_hash_ensure", vec![tb.e()], anys.clone()),
                    s.e(),
                    v.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_rawset",
        &[&tb, &k, &v],
        unit(),
        vec![
            when(is_nil(k.e()), vec![lua_error(text("table index is nil"))]),
            cat.decl(category(k.e())),
            when(
                is_int_cat(&cat),
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
                is_cat(&cat, STR),
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
                is_cat(&cat, FLOAT),
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
                ne(read_global(super::gc::GC_DIRTY, i64()), int(0)),
                vec![expr(call(
                    "zl_gc_before_insert",
                    vec![tb.e(), k.e(), v.e()],
                    unit(),
                ))],
            ),
            expr(call(
                "zb_dict_set",
                vec![
                    call("zl_hash_ensure", vec![tb.e()], anys.clone()),
                    k.e(),
                    v.e(),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));

    // `#t`: the array part's length, a border by construction.
    d.push(define(
        "zl_len",
        &[&tb],
        i64(),
        vec![ret(len(arr_of(tb.e())))],
    ));
    // `#t` with `__len`, as an integer: what the table library
    // measures a table by, which a value the length cannot be read as
    // one refuses.
    d.push(define(
        "zl_table_len",
        &[&tb],
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

    // ─── metatables ─────────────────────────────────────────────
    // What the collector is told once a metatable is set.
    let old_meta = local("old_meta", table.clone());
    let gc_note = |tb: Expr| {
        expr(call(
            "zl_gc_metatable_set",
            vec![tb.clone(), meta_of(tb, t), old_meta.e()],
            unit(),
        ))
    };
    d.push(define(
        "zl_setmetatable",
        &[&tb, &mt],
        table.clone(),
        vec![
            when(
                and(
                    ne(meta_of(tb.e(), t), null(table.clone())),
                    not(is_nil(call(
                        "zl_meta",
                        vec![tb.e(), text("__metatable")],
                        any(),
                    ))),
                ),
                vec![lua_error(text("cannot change a protected metatable"))],
            ),
            old_meta.decl(meta_of(tb.e(), t)),
            if_(
                is_nil(mt.e()),
                vec![set_field(tb.e(), "meta", null(table.clone()))],
                vec![
                    when(
                        not(is_table(mt.e())),
                        vec![lua_error(concat(vec![
                            text("bad argument #2 to 'setmetatable' (nil or table expected, got "),
                            arg_type_name(mt.e()),
                            text(")"),
                        ]))],
                    ),
                    set_field(tb.e(), "meta", unbox_table(mt.e(), t)),
                ],
            ),
            gc_note(tb.e()),
            ret(tb.e()),
        ],
    ));
    // The same for two tables the program holds as such; the metatable
    // null to remove it, the table null for nil, which is the error.
    let mt_table = kept("mt", table.clone());
    d.push(define(
        "zl_setmetatable_tables",
        &[&tb, &mt_table],
        table.clone(),
        vec![
            when(
                eq(tb.e(), null(table.clone())),
                vec![
                    lua_error(text(
                        "bad argument #1 to 'setmetatable' (table expected, got nil)",
                    )),
                    ret(call("zl_table_new", vec![], table.clone())),
                ],
            ),
            when(
                and(
                    ne(meta_of(tb.e(), t), null(table.clone())),
                    not(is_nil(call(
                        "zl_meta",
                        vec![tb.e(), text("__metatable")],
                        any(),
                    ))),
                ),
                vec![lua_error(text("cannot change a protected metatable"))],
            ),
            old_meta.decl(meta_of(tb.e(), t)),
            set_field(tb.e(), "meta", mt_table.e()),
            gc_note(tb.e()),
            ret(tb.e()),
        ],
    ));
    let protected = local("protected", any());
    d.push(define(
        "zl_getmetatable",
        &[&o],
        any(),
        vec![
            when(
                eq(category(o.e()), int(STR)),
                vec![ret(call("zl_string_metatable", vec![], any()))],
            ),
            when(
                is_file(o.e()),
                vec![ret(call("zl_file_metatable", vec![], any()))],
            ),
            when(
                not(is_table(o.e())),
                vec![
                    mt.decl(call("zl_type_meta", vec![o.e()], any())),
                    when(is_nil(mt.e()), vec![ret(nil())]),
                    handler.decl(call(
                        "zl_rawget_str",
                        vec![unbox_table(mt.e(), t), text("__metatable")],
                        any(),
                    )),
                    when(not(is_nil(handler.e())), vec![ret(handler.e())]),
                    ret(mt.e()),
                ],
            ),
            tb.decl(unbox_table(o.e(), t)),
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret(nil())],
            ),
            // `__metatable` in the metatable is what is seen instead.
            protected.decl(call("zl_meta", vec![tb.e(), text("__metatable")], any())),
            when(not(is_nil(protected.e())), vec![ret(protected.e())]),
            ret(box_table(meta_of(tb.e(), t))),
        ],
    ));
    // The metatable's field for an event, or nil.
    let event = kept("event", string());
    d.push(define(
        "zl_meta",
        &[&tb, &event],
        any(),
        vec![
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret(nil())],
            ),
            ret(call(
                "zl_rawget_str",
                vec![meta_of(tb.e(), t), event.e()],
                any(),
            )),
        ],
    ));
    // The event in the metatable of `o`'s type, set by
    // `debug.setmetatable`: nil when none.
    let type_event = |o: &Local| {
        vec![
            mt.decl(call("zl_type_meta", vec![o.e()], any())),
            when(is_nil(mt.e()), vec![ret(nil())]),
            ret(call(
                "zl_rawget_str",
                vec![unbox_table(mt.e(), t), event.e()],
                any(),
            )),
        ]
    };
    // The event of a dynamic value: its metatable's field, nil when it
    // has none. (Strings answer to the string library, which
    // `zl_index` handles itself.)
    d.push(define(
        "zl_meta_of",
        &[&o, &event],
        any(),
        vec![
            when(is_nil(o.e()), type_event(&o)),
            // Strings and files have metatables of their own.
            when(
                eq(category(o.e()), int(STR)),
                vec![ret(call(
                    "zl_rawget_str",
                    vec![
                        unbox_table(call("zl_string_metatable", vec![], any()), t),
                        event.e(),
                    ],
                    any(),
                ))],
            ),
            when(
                is_file(o.e()),
                vec![ret(call(
                    "zl_rawget_str",
                    vec![
                        unbox_table(call("zl_file_metatable", vec![], any()), t),
                        event.e(),
                    ],
                    any(),
                ))],
            ),
            // Any other value that is not a table, its type's.
            when(not(is_table(o.e())), type_event(&o)),
            ret(call(
                "zl_meta",
                vec![unbox_table(o.e(), t), event.e()],
                any(),
            )),
        ],
    ));

    // ─── to-be-closed variables ─────────────────────────────────
    // A `<close>` variable holds nil, false, or a value with `__close`.
    let name = kept("name", string());
    let err = kept("err", any());
    d.push(define(
        "zl_closable",
        &[&o, &name],
        unit(),
        vec![
            when(
                not(call("zl_truthy", vec![o.e()], boolean())),
                vec![ret_void()],
            ),
            when(
                is_nil(call("zl_meta_of", vec![o.e(), text("__close")], any())),
                vec![lua_error(concat(vec![
                    text("variable '"),
                    name.e(),
                    text("' got a non-closable value"),
                ]))],
            ),
            ret_void(),
        ],
    ));
    // Leaving its block: `__close(value, error)`, the error nil on a
    // normal exit.
    d.push(define(
        "zl_close",
        &[&o, &err],
        unit(),
        vec![
            when(
                not(call("zl_truthy", vec![o.e()], boolean())),
                vec![ret_void()],
            ),
            x.decl(call("zl_meta_of", vec![o.e(), text("__close")], any())),
            when(
                is_nil(x.e()),
                vec![lua_error(text("attempt to close non-closable variable"))],
            ),
            metamethod_call_check(x.e(), text("close")),
            // On an error exit the error is in flight: it is put aside
            // while the handler runs, which may raise one of its own
            // that replaces it, and put back otherwise. Closed by
            // `coroutine.close`, the handler sees no error.
            handler.decl(pending()),
            set_global(PENDING, nil()),
            expr(call(
                "zl_call_2",
                vec![x.e(), o.e(), if_expr(is_closing(err.e()), nil(), err.e())],
                any(),
            )),
            when(is_nil(pending()), vec![set_global(PENDING, handler.e())]),
            ret_void(),
        ],
    ));

    // ─── indexing with events ───────────────────────────────────
    // `t[k]` on a table: the raw value, or what `__index` says when
    // that is nil. A table handler is indexed in turn, a function
    // handler is called with the table and the key.
    let index_miss = |raw: Expr, key: Expr| {
        vec![
            x.decl(raw),
            when(not(is_nil(x.e())), vec![ret(x.e())]),
            handler.decl(call("zl_meta", vec![tb.e(), text("__index")], any())),
            when(is_nil(handler.e()), vec![ret(nil())]),
            when(
                is_table(handler.e()),
                vec![ret(call(
                    "zl_table_index",
                    vec![unbox_table(handler.e(), t), key.clone()],
                    any(),
                ))],
            ),
            when(
                is_func(handler.e()),
                vec![
                    metamethod_site(text("index")),
                    ret(call(
                        "zl_first",
                        vec![call(
                            "zl_call_2",
                            vec![handler.e(), box_table(tb.e()), key.clone()],
                            any(),
                        )],
                        any(),
                    )),
                ],
            ),
            x.set(call("zl_index", vec![handler.e(), key], any())),
            when(
                not(is_nil(read_global(PENDING, any()))),
                vec![set_global(VARINFO, int(0))],
            ),
            ret(x.e()),
        ]
    };
    d.push(define(
        "zl_table_index",
        &[&tb, &k],
        any(),
        index_miss(call("zl_rawget", vec![tb.e(), k.e()], any()), k.e()),
    ));
    d.push(define(
        "zl_table_index_str",
        &[&tb, &s],
        any(),
        index_miss(
            call("zl_rawget_str", vec![tb.e(), s.e()], any()),
            box_str(s.e()),
        ),
    ));
    d.push(define(
        "zl_table_geti",
        &[&tb, &i],
        any(),
        index_miss(
            call("zl_rawgeti", vec![tb.e(), i.e()], any()),
            box_i64(i.e()),
        ),
    ));
    d.push(define(
        "zl_table_index_key",
        &[&tb, &k],
        any(),
        index_miss(call("zl_rawget_key", vec![tb.e(), k.e()], any()), k.e()),
    ));
    // `t[k] = v` on a table: the raw store, unless the key is absent
    // and `__newindex` says otherwise.
    let newindex = |present: Expr, store: Stmt, key: Expr| {
        vec![
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![store.clone(), ret_void()],
            ),
            when(not(is_nil(present)), vec![store.clone(), ret_void()]),
            handler.decl(call("zl_meta", vec![tb.e(), text("__newindex")], any())),
            when(is_nil(handler.e()), vec![store, ret_void()]),
            when(
                is_table(handler.e()),
                vec![
                    expr(call(
                        "zl_table_setindex",
                        vec![unbox_table(handler.e(), t), key.clone(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                is_func(handler.e()),
                vec![
                    metamethod_site(text("newindex")),
                    expr(call(
                        "zl_call_3",
                        vec![handler.e(), box_table(tb.e()), key.clone(), v.e()],
                        any(),
                    )),
                    ret_void(),
                ],
            ),
            expr(call("zl_setindex", vec![handler.e(), key, v.e()], unit())),
            when(
                not(is_nil(read_global(PENDING, any()))),
                vec![set_global(VARINFO, int(0))],
            ),
            ret_void(),
        ]
    };
    d.push(define(
        "zl_table_setindex",
        &[&tb, &k, &v],
        unit(),
        newindex(
            call("zl_rawget", vec![tb.e(), k.e()], any()),
            expr(call("zl_rawset", vec![tb.e(), k.e(), v.e()], unit())),
            k.e(),
        ),
    ));
    d.push(define(
        "zl_table_setindex_str",
        &[&tb, &s, &v],
        unit(),
        newindex(
            call("zl_rawget_str", vec![tb.e(), s.e()], any()),
            expr(call("zl_rawset_str", vec![tb.e(), s.e(), v.e()], unit())),
            box_str(s.e()),
        ),
    ));
    d.push(define(
        "zl_table_seti",
        &[&tb, &i, &v],
        unit(),
        newindex(
            call("zl_rawgeti", vec![tb.e(), i.e()], any()),
            expr(call("zl_rawseti", vec![tb.e(), i.e(), v.e()], unit())),
            box_i64(i.e()),
        ),
    ));
    d.push(define(
        "zl_table_setindex_key",
        &[&tb, &k, &v],
        unit(),
        newindex(
            call("zl_rawget_key", vec![tb.e(), k.e()], any()),
            expr(call("zl_rawset_key", vec![tb.e(), k.e(), v.e()], unit())),
            k.e(),
        ),
    ));

    // The same on a dynamic value: a table is indexed as above, a
    // string through the string library, anything else is an error.
    // A value of another type is indexed through its type's metatable
    // (`debug.setmetatable`), when it has one with the event.
    let mm = kept("mm", any());
    let type_index = |o: &Local, key: Expr| {
        block_of(vec![
            mm.decl(call("zl_meta_of", vec![o.e(), text("__index")], any())),
            when(
                not(is_nil(mm.e())),
                vec![
                    when(
                        is_func(mm.e()),
                        vec![
                            metamethod_site(text("index")),
                            ret(call(
                                "zl_first",
                                vec![call("zl_call_2", vec![mm.e(), o.e(), key.clone()], any())],
                                any(),
                            )),
                        ],
                    ),
                    ret(call("zl_index", vec![mm.e(), key], any())),
                ],
            ),
        ])
    };
    let type_newindex = |o: &Local, key: Expr| {
        block_of(vec![
            mm.decl(call("zl_meta_of", vec![o.e(), text("__newindex")], any())),
            when(
                not(is_nil(mm.e())),
                vec![
                    if_(
                        is_func(mm.e()),
                        vec![
                            metamethod_site(text("newindex")),
                            expr(call(
                                "zl_call_3",
                                vec![mm.e(), o.e(), key.clone(), v.e()],
                                any(),
                            )),
                        ],
                        vec![expr(call("zl_setindex", vec![mm.e(), key, v.e()], unit()))],
                    ),
                    ret_void(),
                ],
            ),
        ])
    };
    let not_indexable = |o: &Local| {
        type_error(
            concat(vec![
                text("attempt to index a "),
                obj_type_name(o.e()),
                text(" value"),
            ]),
            int(OPERAND_LEFT),
        )
    };
    d.push(define(
        "zl_index",
        &[&o, &k],
        any(),
        vec![
            when(
                is_table(o.e()),
                vec![ret(call(
                    "zl_table_index",
                    vec![unbox_table(o.e(), t), k.e()],
                    any(),
                ))],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![ret(call("zl_foreign_index", vec![o.e(), k.e()], any()))],
            ),
            when(
                eq(category(o.e()), int(STR)),
                vec![ret(call("zl_string_member", vec![o.e(), k.e()], any()))],
            ),
            when(
                is_file(o.e()),
                vec![ret(call("zl_file_member", vec![o.e(), k.e()], any()))],
            ),
            type_index(&o, k.e()),
            not_indexable(&o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_index_str",
        &[&o, &s],
        any(),
        vec![
            when(
                is_table(o.e()),
                vec![ret(call(
                    "zl_table_index_str",
                    vec![unbox_table(o.e(), t), s.e()],
                    any(),
                ))],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![ret(call("zb_foreign_get", vec![o.e(), s.e()], any()))],
            ),
            when(
                eq(category(o.e()), int(STR)),
                vec![ret(call(
                    "zl_string_member",
                    vec![o.e(), box_str(s.e())],
                    any(),
                ))],
            ),
            when(
                is_file(o.e()),
                vec![ret(call(
                    "zl_file_member",
                    vec![o.e(), box_str(s.e())],
                    any(),
                ))],
            ),
            type_index(&o, box_str(s.e())),
            not_indexable(&o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_index_key",
        &[&o, &k],
        any(),
        vec![
            when(
                is_table(o.e()),
                vec![ret(call(
                    "zl_table_index_key",
                    vec![unbox_table(o.e(), t), k.e()],
                    any(),
                ))],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![ret(call("zl_foreign_index", vec![o.e(), k.e()], any()))],
            ),
            when(
                eq(category(o.e()), int(STR)),
                vec![ret(call("zl_string_member", vec![o.e(), k.e()], any()))],
            ),
            when(
                is_file(o.e()),
                vec![ret(call("zl_file_member", vec![o.e(), k.e()], any()))],
            ),
            type_index(&o, k.e()),
            not_indexable(&o),
            ret(nil()),
        ],
    ));
    // Indexing nil where the frontend knows the receiver is nil.
    d.push(define_cold(
        "zl_index_nil",
        &[],
        any(),
        vec![
            type_error(text("attempt to index a nil value"), int(OPERAND_LEFT)),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_setindex_key",
        &[&o, &k, &v],
        unit(),
        vec![
            when(
                is_table(o.e()),
                vec![
                    expr(call(
                        "zl_table_setindex_key",
                        vec![unbox_table(o.e(), t), k.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![
                    expr(call(
                        "zl_foreign_setindex",
                        vec![o.e(), k.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            not_indexable(&o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_geti",
        &[&o, &i],
        any(),
        vec![
            when(
                is_table(o.e()),
                vec![ret(call(
                    "zl_table_geti",
                    vec![unbox_table(o.e(), t), i.e()],
                    any(),
                ))],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![ret(call(
                    "zl_foreign_index",
                    vec![o.e(), box_i64(i.e())],
                    any(),
                ))],
            ),
            when(eq(category(o.e()), int(STR)), vec![ret(nil())]),
            type_index(&o, box_i64(i.e())),
            not_indexable(&o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_setindex",
        &[&o, &k, &v],
        unit(),
        vec![
            when(
                is_table(o.e()),
                vec![
                    expr(call(
                        "zl_table_setindex",
                        vec![unbox_table(o.e(), t), k.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![
                    expr(call(
                        "zl_foreign_setindex",
                        vec![o.e(), k.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            type_newindex(&o, k.e()),
            not_indexable(&o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_setindex_str",
        &[&o, &s, &v],
        unit(),
        vec![
            when(
                is_table(o.e()),
                vec![
                    expr(call(
                        "zl_table_setindex_str",
                        vec![unbox_table(o.e(), t), s.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![
                    expr(call("zb_foreign_set", vec![o.e(), s.e(), v.e()], unit())),
                    ret_void(),
                ],
            ),
            type_newindex(&o, box_str(s.e())),
            not_indexable(&o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_seti",
        &[&o, &i, &v],
        unit(),
        vec![
            when(
                is_table(o.e()),
                vec![
                    expr(call(
                        "zl_table_seti",
                        vec![unbox_table(o.e(), t), i.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                zyntax_builtins::foreign::is_foreign(o.e()),
                vec![
                    expr(call(
                        "zl_foreign_setindex",
                        vec![o.e(), box_i64(i.e()), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            type_newindex(&o, box_i64(i.e())),
            not_indexable(&o),
            ret_void(),
        ],
    ));
    // The table a dynamic value must be, for the raw functions. When it
    // is not one, the error is raised and an empty table stands in, so
    // whatever runs before the error is seen has a table to run on.
    let what = kept("what", string());
    d.push(define(
        "zl_as_table",
        &[&o, &what],
        table.clone(),
        vec![
            when(
                not(is_table(o.e())),
                vec![
                    lua_error(concat(vec![
                        what.e(),
                        text(" (table expected, got "),
                        arg_type_name(o.e()),
                        text(")"),
                    ])),
                    ret(call("zl_table_new", vec![], table.clone())),
                ],
            ),
            ret(unbox_table(o.e(), t)),
        ],
    ));

    // ─── traversal ──────────────────────────────────────────────
    // A traversal position: `0..n` for the array part, `n + j` for
    // slot `j` of a shaped table, `n + slots + e` for entry `e` of the
    // hash part. The next position at or after `pos` holding a value,
    // or -1 when the traversal is over.
    let pos = local("pos", i64());
    let count = local("count", i64());
    let slots = local("slots", i64());
    d.push(define(
        "zl_next_pos",
        &[&tb, &pos],
        i64(),
        vec![
            arr.decl(arr_of(tb.e())),
            n.decl(len(arr.e())),
            i.decl(pos.e()),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(not(is_nil(at(arr.e(), i.e()))), vec![ret(i.e())]),
                    i.add_assign(int(1)),
                ],
            ),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            while_(
                lt(sub(i.e(), n.e()), slots.e()),
                vec![
                    when(
                        super::slot_present(tb.e(), sub(i.e(), n.e())),
                        vec![ret(i.e())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(int(-1))]),
            h.decl(hash_of(tb.e())),
            count.decl(call("zb_dict_len", vec![h.e()], i64())),
            while_(
                lt(sub(sub(i.e(), n.e()), slots.e()), count.e()),
                vec![
                    when(
                        not(is_nil(at(
                            h.e(),
                            add(mul(sub(sub(i.e(), n.e()), slots.e()), int(2)), int(2)),
                        ))),
                        vec![ret(i.e())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    d.push(define(
        "zl_pos_key",
        &[&tb, &pos],
        any(),
        vec![
            n.decl(len(arr_of(tb.e()))),
            when(lt(pos.e(), n.e()), vec![ret(box_i64(add(pos.e(), int(1))))]),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            when(
                lt(sub(pos.e(), n.e()), slots.e()),
                vec![ret(box_str(slot_key(tb.e(), sub(pos.e(), n.e()))))],
            ),
            ret(at(
                hash_of(tb.e()),
                add(mul(sub(sub(pos.e(), n.e()), slots.e()), int(2)), int(1)),
            )),
        ],
    ));
    d.push(define(
        "zl_pos_value",
        &[&tb, &pos],
        any(),
        vec![
            arr.decl(arr_of(tb.e())),
            n.decl(len(arr.e())),
            when(lt(pos.e(), n.e()), vec![ret(at(arr.e(), pos.e()))]),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            when(
                lt(sub(pos.e(), n.e()), slots.e()),
                vec![ret(slot_load(tb.e(), sub(pos.e(), n.e())))],
            ),
            ret(at(
                hash_of(tb.e()),
                add(mul(sub(sub(pos.e(), n.e()), slots.e()), int(2)), int(2)),
            )),
        ],
    ));
    // The position just past key `k`: 0 for nil, the array slot for an
    // integer in the array part, the hash entry for anything else.
    d.push(define(
        "zl_pos_after",
        &[&tb, &k],
        i64(),
        vec![
            when(is_nil(k.e()), vec![ret(int(0))]),
            n.decl(len(arr_of(tb.e()))),
            cat.decl(category(k.e())),
            when(
                is_int_cat(&cat),
                vec![
                    i.decl(get_i64(k.e())),
                    when(and(ge(i.e(), int(1)), le(i.e(), n.e())), vec![ret(i.e())]),
                ],
            ),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            // A string naming a slot: the position after that slot.
            when(
                and(gt(slots.e(), int(0)), eq(cat.e(), int(STR))),
                vec![
                    slot.decl(slot_index(tb.e(), get_str(k.e()))),
                    when(
                        ge(slot.e(), int(0)),
                        vec![ret(add(add(n.e(), slot.e()), int(1)))],
                    ),
                ],
            ),
            when(
                not(is_nil(hash_field(tb.e()))),
                vec![
                    h.decl(hash_of(tb.e())),
                    i.decl(int(0)),
                    count.decl(call("zb_dict_len", vec![h.e()], i64())),
                    while_(
                        lt(i.e(), count.e()),
                        vec![
                            when(
                                call(
                                    "zb_any_eq",
                                    vec![at(h.e(), add(mul(i.e(), int(2)), int(1))), k.e()],
                                    boolean(),
                                ),
                                vec![ret(add(add(add(n.e(), slots.e()), i.e()), int(1)))],
                            ),
                            i.add_assign(int(1)),
                        ],
                    ),
                ],
            ),
            // An index past the array part but within its longest
            // was an element removed during the traversal, which
            // shortened the part: the traversal goes on from its end.
            when(
                and(
                    is_int_cat(&cat),
                    and(
                        ge(get_i64(k.e()), int(1)),
                        le(get_i64(k.e()), super::high_of(tb.e())),
                    ),
                ),
                vec![ret(n.e())],
            ),
            lua_error(text("invalid key to 'next'")),
            ret(int(-1)),
        ],
    ));
    // The position after `pos` when the array part held `seen` elements
    // at the last step: elements removed since moved what follows.
    let seen = local("seen", i64());
    d.push(define(
        "zl_next_pos_from",
        &[&tb, &pos, &seen],
        i64(),
        vec![
            n.decl(len(arr_of(tb.e()))),
            i.decl(pos.e()),
            when(
                lt(n.e(), seen.e()),
                vec![if_(
                    ge(i.e(), seen.e()),
                    vec![i.set(sub(i.e(), sub(seen.e(), n.e())))],
                    vec![when(gt(i.e(), n.e()), vec![i.set(n.e())])],
                )],
            ),
            ret(call("zl_next_pos", vec![tb.e(), i.e()], i64())),
        ],
    ));
    // `next(t, k)`: the key and value after `k`, as two values, or
    // nothing.
    let found = local("found", any());
    d.push(define(
        "zl_next",
        &[&tb, &k],
        anys.clone(),
        vec![
            pos.decl(call(
                "zl_next_pos",
                vec![tb.e(), call("zl_pos_after", vec![tb.e(), k.e()], i64())],
                i64(),
            )),
            // The traversal ends with one nil.
            when(
                lt(pos.e(), int(0)),
                vec![ret(list(vec![nil()], anys.clone()))],
            ),
            // The value is read first and held: making the key may
            // collect, which clears a weak value nothing holds.
            found.decl(call("zl_pos_value", vec![tb.e(), pos.e()], any())),
            ret(list(
                vec![call("zl_pos_key", vec![tb.e(), pos.e()], any()), found.e()],
                anys.clone(),
            )),
        ],
    ));
    d
}
