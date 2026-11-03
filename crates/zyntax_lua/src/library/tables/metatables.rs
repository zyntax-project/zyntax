//! Metatables declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        handler,
        mt,
        o,
        table,
        tb,
        x,
        ..
    } = context;

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
        &[tb, mt],
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
        &[tb, &mt_table],
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
        &[o],
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
                    mt.decl(if_expr(
                        is_userdata(o.e()),
                        call("zl_ud_meta", vec![o.e()], any()),
                        call("zl_type_meta", vec![o.e()], any()),
                    )),
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
        &[tb, &event],
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
        &[o, &event],
        any(),
        vec![
            when(is_nil(o.e()), type_event(o)),
            // Strings and files have metatables of their own.
            when(
                eq(category(o.e()), int(STR)),
                vec![
                    mt.decl(read_global("zl_string_meta", any())),
                    // The default string metatable only has `__index`,
                    // which string indexing handles directly. Avoid
                    // constructing the full string library for unrelated
                    // metamethod probes such as numeric `tostring`.
                    when(is_nil(mt.e()), vec![ret(nil())]),
                    ret(call(
                        "zl_rawget_str",
                        vec![unbox_table(mt.e(), t), event.e()],
                        any(),
                    )),
                ],
            ),
            when(
                is_file(o.e()),
                vec![
                    mt.decl(read_global("zl_file_meta", any())),
                    when(is_nil(mt.e()), vec![ret(nil())]),
                    ret(call(
                        "zl_rawget_str",
                        vec![unbox_table(mt.e(), t), event.e()],
                        any(),
                    )),
                ],
            ),
            // Any other value that is not a table, its type's; a full
            // userdata, its own.
            when(
                not(is_table(o.e())),
                vec![
                    when(
                        is_userdata(o.e()),
                        vec![
                            mt.decl(call("zl_ud_meta", vec![o.e()], any())),
                            when(is_nil(mt.e()), vec![ret(nil())]),
                            ret(call(
                                "zl_rawget_str",
                                vec![unbox_table(mt.e(), t), event.e()],
                                any(),
                            )),
                        ],
                    ),
                    block_of(type_event(o)),
                ],
            ),
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
    d.push(define(
        "zl_closable",
        &[o, &name],
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
    // Leaving its block: `__close(value, error)`, the error the one in
    // flight or nil. It is put aside while the handler runs, which may
    // raise one of its own that replaces it, and put back otherwise.
    // Closed by `coroutine.close`, the handler sees no error. While
    // the state closes for `os.exit`, only the main thread's variables
    // are closed, each seeing the error the last handler raised, and
    // nothing a handler raises stops the exit.
    d.push(define(
        "zl_close",
        &[o],
        unit(),
        vec![
            when(
                not(call("zl_truthy", vec![o.e()], boolean())),
                vec![ret_void()],
            ),
            handler.decl(pending()),
            when(
                is_exiting(handler.e()),
                vec![
                    when(
                        is_nil(read_global(crate::library::coroutines::CURRENT, any())),
                        vec![
                            set_global(PENDING, read_global(EXIT_ERROR, any())),
                            expr(call("zl_close", vec![o.e()], unit())),
                            set_global(EXIT_ERROR, pending()),
                        ],
                    ),
                    set_global(PENDING, handler.e()),
                    ret_void(),
                ],
            ),
            set_global(PENDING, nil()),
            x.decl(call("zl_meta_of", vec![o.e(), text("__close")], any())),
            // Closed while unwinding, the handler is called from no Lua
            // frame: the message has no position and names no event.
            when(
                and(
                    not(is_nil(handler.e())),
                    and(
                        not(is_func(x.e())),
                        is_nil(call("zl_meta_of", vec![x.e(), text("__call")], any())),
                    ),
                ),
                vec![
                    expr(call(
                        "zl_raise_value",
                        vec![box_str(concat(vec![
                            text("attempt to call a "),
                            type_name(x.e()),
                            text(" value"),
                        ]))],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            metamethod_call_check(x.e(), text("close")),
            expr(call(
                "zl_call_2",
                vec![
                    x.e(),
                    o.e(),
                    if_expr(
                        or(is_closing(handler.e()), is_nil_error(handler.e())),
                        nil(),
                        handler.e(),
                    ),
                ],
                any(),
            )),
            when(is_nil(pending()), vec![set_global(PENDING, handler.e())]),
            ret_void(),
        ],
    ));
}
