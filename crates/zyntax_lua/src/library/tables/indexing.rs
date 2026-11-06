//! Indexing declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        handler,
        i,
        k,
        n,
        o,
        s,
        table,
        tb,
        v,
        x,
        ..
    } = context;

    // ─── indexing with events ───────────────────────────────────
    // `t[k]` on a table: the raw value, or what `__index` says when
    // that is nil. A function handler is called with the table and the
    // key; any other handler is indexed in turn, by the chain below.
    let index_miss = |raw: Expr, key: Expr| {
        vec![
            x.decl(raw),
            when(not(is_nil(x.e())), vec![ret(x.e())]),
            handler.decl(call("zl_meta", vec![tb.e(), text("__index")], any())),
            when(is_nil(handler.e()), vec![ret(nil())]),
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
            ret(call("zl_index_chain", vec![handler.e(), key], any())),
        ]
    };
    // The handlers after the first, as a loop: `cur` is the value the
    // last handler named and `n` how many handlers have been looked up.
    // A table is read raw before its own handler is looked up; a value
    // with no handler of its own is indexed as itself, which is an
    // error for most. The reference gives up after `MAXTAGLOOP`.
    let cur = kept("cur", any());
    let raw_of = local("raw_of", table.clone());
    let raw_get = || call("zl_rawget", vec![raw_of.e(), k.e()], any());
    let raw_set = || expr(call("zl_rawset", vec![raw_of.e(), k.e(), v.e()], unit()));
    let too_long = |event: &str| {
        when(
            ge(n.e(), int(MAXTAGLOOP)),
            vec![lua_error(text(&format!(
                "'{event}' chain too long; possible loop"
            )))],
        )
    };
    d.push(define(
        "zl_index_chain",
        &[o, k],
        any(),
        vec![
            cur.decl(o.e()),
            n.decl(int(1)),
            x.decl(nil()),
            handler.decl(nil()),
            while_(
                le(n.e(), int(MAXTAGLOOP)),
                vec![
                    if_(
                        is_table(cur.e()),
                        vec![
                            raw_of.decl(unbox_table(cur.e(), t)),
                            x.set(raw_get()),
                            when(not(is_nil(x.e())), vec![ret(x.e())]),
                            too_long("__index"),
                            handler.set(call("zl_meta", vec![raw_of.e(), text("__index")], any())),
                            when(is_nil(handler.e()), vec![ret(nil())]),
                        ],
                        vec![
                            too_long("__index"),
                            handler.set(if_expr(
                                or(eq(category(cur.e()), int(STR)), is_file(cur.e())),
                                nil(),
                                call("zl_meta_of", vec![cur.e(), text("__index")], any()),
                            )),
                            when(
                                is_nil(handler.e()),
                                vec![
                                    x.set(call("zl_index", vec![cur.e(), k.e()], any())),
                                    when(
                                        not(is_nil(read_global(PENDING, any()))),
                                        vec![set_global(VARINFO, int(0))],
                                    ),
                                    ret(x.e()),
                                ],
                            ),
                        ],
                    ),
                    when(
                        is_func(handler.e()),
                        vec![
                            metamethod_site(text("index")),
                            ret(call(
                                "zl_first",
                                vec![call("zl_call_2", vec![handler.e(), cur.e(), k.e()], any())],
                                any(),
                            )),
                        ],
                    ),
                    cur.set(handler.e()),
                    n.add_assign(int(1)),
                ],
            ),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_table_index",
        &[tb, k],
        any(),
        index_miss(call("zl_rawget", vec![tb.e(), k.e()], any()), k.e()),
    ));
    d.push(define(
        "zl_table_index_str",
        &[tb, s],
        any(),
        index_miss(
            call("zl_rawget_str", vec![tb.e(), s.e()], any()),
            box_str(s.e()),
        ),
    ));
    d.push(define(
        "zl_table_geti",
        &[tb, i],
        any(),
        index_miss(
            call("zl_rawgeti", vec![tb.e(), i.e()], any()),
            box_i64(i.e()),
        ),
    ));
    d.push(define(
        "zl_table_index_key",
        &[tb, k],
        any(),
        index_miss(call("zl_rawget_key", vec![tb.e(), k.e()], any()), k.e()),
    ));
    // `t[k] = v` on a table: the raw store, unless the key is absent
    // and `__newindex` says otherwise. A function handler is called
    // with the table, the key and the value; any other handler is
    // stored into in turn, by the chain below.
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
            expr(call(
                "zl_newindex_chain",
                vec![handler.e(), key, v.e()],
                unit(),
            )),
            ret_void(),
        ]
    };
    // The handlers after the first, as [`zl_index_chain`] walks them:
    // a table with the key present, or with no handler of its own, is
    // stored into raw. The bound is checked before a table without a
    // metatable is stored into, as the reference checks it.
    d.push(define(
        "zl_newindex_chain",
        &[o, k, v],
        unit(),
        vec![
            cur.decl(o.e()),
            n.decl(int(1)),
            handler.decl(nil()),
            while_(
                le(n.e(), int(MAXTAGLOOP)),
                vec![
                    if_(
                        is_table(cur.e()),
                        vec![
                            raw_of.decl(unbox_table(cur.e(), t)),
                            when(not(is_nil(raw_get())), vec![raw_set(), ret_void()]),
                            too_long("__newindex"),
                            handler.set(call(
                                "zl_meta",
                                vec![raw_of.e(), text("__newindex")],
                                any(),
                            )),
                            when(is_nil(handler.e()), vec![raw_set(), ret_void()]),
                        ],
                        vec![
                            too_long("__newindex"),
                            handler.set(call(
                                "zl_meta_of",
                                vec![cur.e(), text("__newindex")],
                                any(),
                            )),
                            when(
                                is_nil(handler.e()),
                                vec![
                                    expr(call("zl_setindex", vec![cur.e(), k.e(), v.e()], unit())),
                                    when(
                                        not(is_nil(read_global(PENDING, any()))),
                                        vec![set_global(VARINFO, int(0))],
                                    ),
                                    ret_void(),
                                ],
                            ),
                        ],
                    ),
                    when(
                        is_func(handler.e()),
                        vec![
                            metamethod_site(text("newindex")),
                            expr(call(
                                "zl_call_3",
                                vec![handler.e(), cur.e(), k.e(), v.e()],
                                any(),
                            )),
                            ret_void(),
                        ],
                    ),
                    cur.set(handler.e()),
                    n.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_setindex",
        &[tb, k, v],
        unit(),
        newindex(
            call("zl_rawget", vec![tb.e(), k.e()], any()),
            expr(call("zl_rawset", vec![tb.e(), k.e(), v.e()], unit())),
            k.e(),
        ),
    ));
    d.push(define(
        "zl_table_setindex_str",
        &[tb, s, v],
        unit(),
        newindex(
            call("zl_rawget_str", vec![tb.e(), s.e()], any()),
            expr(call("zl_rawset_str", vec![tb.e(), s.e(), v.e()], unit())),
            box_str(s.e()),
        ),
    ));
    d.push(define(
        "zl_table_seti",
        &[tb, i, v],
        unit(),
        newindex(
            call("zl_rawgeti", vec![tb.e(), i.e()], any()),
            expr(call("zl_rawseti", vec![tb.e(), i.e(), v.e()], unit())),
            box_i64(i.e()),
        ),
    ));
    // `t[i] = v` for an integer value, which a flat array of integers
    // takes without a box: in place when the element is there, at the
    // end or as the first element of an empty array when no metatable
    // or hash part could say otherwise. Anything else stores the box.
    let vi = local("vi", i64());
    let flat = local("flat", t.ints());
    let flat_len = local("flat_len", i64());
    d.push(define(
        "zl_table_seti_int",
        &[tb, i, &vi],
        unit(),
        vec![
            when(
                crate::library::arr_is_ints(tb.e()),
                vec![
                    flat.decl(crate::library::arr_ints(tb.e(), t)),
                    flat_len.decl(len(flat.e())),
                    when(
                        and(ge(i.e(), int(1)), le(i.e(), flat_len.e())),
                        vec![set_idx(flat.e(), sub(i.e(), int(1)), vi.e()), ret_void()],
                    ),
                    when(
                        and(
                            eq(i.e(), add(flat_len.e(), int(1))),
                            and(
                                eq(meta_of(tb.e(), t), null(table.clone())),
                                is_nil(hash_field(tb.e())),
                            ),
                        ),
                        vec![push(flat.e(), vi.e()), ret_void()],
                    ),
                ],
            ),
            when(
                and(
                    eq(i.e(), int(1)),
                    and(
                        eq(
                            crate::library::arr_field(tb.e()),
                            read_global(ARR_EMPTY, any()),
                        ),
                        and(
                            eq(meta_of(tb.e(), t), null(table.clone())),
                            is_nil(hash_field(tb.e())),
                        ),
                    ),
                ),
                vec![
                    set_field(
                        tb.e(),
                        "arr",
                        call("zb_list_box_i64", vec![list(vec![vi.e()], t.ints())], any()),
                    ),
                    ret_void(),
                ],
            ),
            // No metatable: the raw store; otherwise `__newindex` may
            // have a say.
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![
                    expr(call(
                        "zl_rawseti",
                        vec![tb.e(), i.e(), box_i64(vi.e())],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            expr(call(
                "zl_table_seti",
                vec![tb.e(), i.e(), box_i64(vi.e())],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_table_setindex_key",
        &[tb, k, v],
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
                    ret(call("zl_index_chain", vec![mm.e(), key], any())),
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
                        vec![expr(call(
                            "zl_newindex_chain",
                            vec![mm.e(), key, v.e()],
                            unit(),
                        ))],
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
        &[o, k],
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
            type_index(o, k.e()),
            not_indexable(o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_index_str",
        &[o, s],
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
            type_index(o, box_str(s.e())),
            not_indexable(o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_index_key",
        &[o, k],
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
            type_index(o, k.e()),
            not_indexable(o),
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
        &[o, k, v],
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
            not_indexable(o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_geti",
        &[o, i],
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
            type_index(o, box_i64(i.e())),
            not_indexable(o),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_setindex",
        &[o, k, v],
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
            type_newindex(o, k.e()),
            not_indexable(o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_setindex_str",
        &[o, s, v],
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
            type_newindex(o, box_str(s.e())),
            not_indexable(o),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_seti",
        &[o, i, v],
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
            type_newindex(o, box_i64(i.e())),
            not_indexable(o),
            ret_void(),
        ],
    ));
    // The table a dynamic value must be, for the raw functions. When it
    // is not one, the error is raised and an empty table stands in, so
    // whatever runs before the error is seen has a table to run on.
    let what = kept("what", string());
    d.push(define(
        "zl_as_table",
        &[o, &what],
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
}
