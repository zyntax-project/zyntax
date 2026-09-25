//! Garbage collection as a program sees it: what `setmetatable` tells
//! the host about weak and finalizable tables, `collectgarbage`, the
//! finalizer runner, the compaction of weak tables a collection
//! changed, and `warn`.

use super::*;

/// How many weak tables a collection changed and nothing compacted
/// since, as the host last reported it; a store of a new key into the
/// hash part of a table compacts that table first while this is not
/// zero.
pub const GC_DIRTY: &str = "zl_gc_dirty";

/// `collectgarbage` options, by the number the host knows them by.
pub const GC_OPTIONS: [&str; 10] = [
    "collect",
    "step",
    "count",
    "isrunning",
    "stop",
    "restart",
    "incremental",
    "generational",
    "setpause",
    "setstepmul",
];

pub fn option(name: &str) -> i64 {
    GC_OPTIONS
        .iter()
        .position(|o| *o == name)
        .expect("a collectgarbage option") as i64
}

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let tb = kept("t", table.clone());
    let mt = kept("mt", table.clone());
    let x = kept("x", any());
    let y = local("y", any());
    let e = local("e", any());
    let s = local("s", string());
    let f = local("f", i64());
    let n = local("n", i64());
    let i = local("i", i64());
    let p = local("p", i64());
    let op = local("op", i64());
    let r = local("r", i64());
    let h = borrowed("h", anys.clone());
    let nd = local("nd", anys.clone());
    let saved = local("saved", any());
    let line = local("line", i64());
    let varinfo = local("varinfo", i64());
    let mut d = vec![
        extern_fn(
            "zl_gc",
            &[("op", i64()), ("arg", i64())],
            i64(),
            Some("$Lua$gc"),
        ),
        extern_fn(
            "zl_gc_note",
            &[("t", i64()), ("flags", i64()), ("runner", usize())],
            unit(),
            Some("$Lua$gc_note"),
        ),
        extern_fn(
            "zl_gc_next_finalizer",
            &[],
            i64(),
            Some("$Lua$gc_next_finalizer"),
        ),
        extern_fn(
            "zl_gc_finalizing",
            &[("on", i64())],
            i64(),
            Some("$Lua$gc_finalizing"),
        ),
        extern_fn("zl_gc_close", &[], i64(), Some("$Lua$gc_close")),
        extern_fn("zl_gc_dirty_count", &[], i64(), Some("$Lua$gc_dirty_count")),
        extern_fn(
            "zl_gc_take_dirty",
            &[("t", i64())],
            i64(),
            Some("$Lua$gc_take_dirty"),
        ),
        global_var(GC_DIRTY, i64()),
        extern_fn(
            "zl_warn_piece",
            &[("message", string()), ("tocont", boolean())],
            unit(),
            Some("$Lua$warn"),
        ),
    ];
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };

    // What a metatable asks of the collector for the tables it is set
    // on: `__mode` naming weak keys (1) or weak values (2), `__gc`
    // present (4). Read when the metatable is set, as the reference
    // reads `__gc`.
    let mode = local("mode", any());
    d.push(define(
        "zl_gc_flags",
        &[&mt],
        i64(),
        vec![
            f.decl(int(0)),
            mode.decl(call("zl_rawget_str", vec![mt.e(), text("__mode")], any())),
            when(
                eq(category(mode.e()), int(STR)),
                vec![
                    s.decl(get_str(mode.e())),
                    when(
                        call("zb_str_contains", vec![s.e(), text("k")], boolean()),
                        vec![f.set(bitor(f.e(), int(1)))],
                    ),
                    when(
                        call("zb_str_contains", vec![s.e(), text("v")], boolean()),
                        vec![f.set(bitor(f.e(), int(2)))],
                    ),
                ],
            ),
            when(
                not(is_nil(call(
                    "zl_rawget_str",
                    vec![mt.e(), text("__gc")],
                    any(),
                ))),
                vec![f.set(bitor(f.e(), int(4)))],
            ),
            ret(f.e()),
        ],
    ));
    // `t` has just been given `mt` (null for none) in place of `old`:
    // the host learns of a table that becomes weak or finalizable, and
    // of one that may stop being weak.
    let old = kept("old", table.clone());
    d.push(define(
        "zl_gc_metatable_set",
        &[&tb, &mt, &old],
        unit(),
        vec![
            f.decl(if_expr(
                eq(mt.e(), null(table.clone())),
                int(0),
                call("zl_gc_flags", vec![mt.e()], i64()),
            )),
            // A dirty table that stops being weak is no longer counted.
            when(
                or(ne(f.e(), int(0)), ne(old.e(), null(table.clone()))),
                vec![
                    expr(call(
                        "zl_gc_note",
                        vec![cast(tb.e(), i64()), f.e(), code_of("zl_gc_finalize")],
                        unit(),
                    )),
                    set_global(GC_DIRTY, call("zl_gc_dirty_count", vec![], i64())),
                ],
            ),
            ret_void(),
        ],
    ));

    // Every finalizer due, each called with its object through the
    // object's metatable as it stands now. An error in one is a
    // warning; the error in flight when they started, and the
    // position it carries, are the same after them.
    let msg_of = |e: Expr| {
        if_expr(
            eq(category(e.clone()), int(STR)),
            get_str(e),
            text("error object is not a string"),
        )
    };
    let piece =
        |text_: Expr, more: bool| expr(call("zl_warn_piece", vec![text_, bool(more)], unit()));
    // One finalizer called on its object; a finalizer that cannot be
    // called is an error naming the event.
    let gc_fn = kept("gc", any());
    let obj = kept("o", any());
    d.push(define(
        "zl_gc_call_finalizer",
        &[&gc_fn, &obj],
        unit(),
        vec![
            metamethod_call_check(gc_fn.e(), text("__gc")),
            expr(call("zl_call_1", vec![gc_fn.e(), obj.e()], any())),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_gc_finalize",
        &[],
        unit(),
        vec![
            set_global(GC_DIRTY, call("zl_gc_dirty_count", vec![], i64())),
            when(
                ne(call("zl_gc_finalizing", vec![int(1)], i64()), int(0)),
                vec![ret_void()],
            ),
            saved.decl(pending()),
            line.decl(read_global(LINE, i64())),
            varinfo.decl(read_global(VARINFO, i64())),
            set_global(PENDING, nil()),
            p.decl(call("zl_gc_next_finalizer", vec![], i64())),
            while_(
                ne(p.e(), int(0)),
                vec![
                    tb.decl(cast(p.e(), table.clone())),
                    y.decl(call("zl_meta", vec![tb.e(), text("__gc")], any())),
                    when(
                        not(is_nil(y.e())),
                        vec![
                            expr(call(
                                "zl_gc_call_finalizer",
                                vec![y.e(), box_table(tb.e())],
                                unit(),
                            )),
                            when(
                                not(is_nil(pending())),
                                vec![
                                    e.decl(call("zl_take_pending", vec![], any())),
                                    piece(text("error in "), true),
                                    piece(text("__gc"), true),
                                    piece(text(" ("), true),
                                    piece(msg_of(e.e()), true),
                                    piece(text(")"), false),
                                ],
                            ),
                        ],
                    ),
                    p.set(call("zl_gc_next_finalizer", vec![], i64())),
                ],
            ),
            set_global(PENDING, saved.e()),
            set_global(LINE, line.e()),
            set_global(VARINFO, varinfo.e()),
            expr(call("zl_gc_finalizing", vec![int(0)], i64())),
            ret_void(),
        ],
    ));

    // A weak table's dict rebuilt without the tombstones a collection
    // left for its dead keys, or the entries whose values are nil.
    let dead = int(zyntax_builtins::instance_tag(DEAD_KEY_KIND));
    d.push(define(
        "zl_gc_compact",
        &[&tb],
        unit(),
        vec![
            when(is_nil(hash_field(tb.e())), vec![ret_void()]),
            h.decl(hash_of(tb.e(), t)),
            nd.decl(call("zb_dict_new", vec![], anys.clone())),
            n.decl(call("zb_dict_len", vec![h.e()], i64())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    y.decl(at(h.e(), add(mul(i.e(), int(2)), int(1)))),
                    e.decl(at(h.e(), add(mul(i.e(), int(2)), int(2)))),
                    when(
                        and(ne(tag_of(y.e()), dead.clone()), not(is_nil(e.e()))),
                        vec![expr(call(
                            "zb_dict_insert",
                            vec![nd.e(), y.e(), e.e()],
                            unit(),
                        ))],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            set_field(tb.e(), "hash", call("zb_list_box_any", vec![nd.e()], any())),
            ret_void(),
        ],
    ));
    // Before a store into `t`'s hash part while some weak table is
    // dirty: a store of a new key (one absent, or present with a nil
    // value) into a dirty table compacts it first. A store to a key
    // present leaves the positions alone, so a traversal in progress
    // goes on.
    let k = kept("k", any());
    let v = kept("v", any());
    let may_compact = |present: Expr| {
        vec![
            when(is_nil(v.e()), vec![ret_void()]),
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret_void()],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret_void()]),
            when(not(is_nil(present)), vec![ret_void()]),
            when(
                eq(
                    call("zl_gc_take_dirty", vec![cast(tb.e(), i64())], i64()),
                    int(0),
                ),
                vec![ret_void()],
            ),
            expr(call("zl_gc_compact", vec![tb.e()], unit())),
            set_global(GC_DIRTY, call("zl_gc_dirty_count", vec![], i64())),
            ret_void(),
        ]
    };
    d.push(define(
        "zl_gc_before_insert",
        &[&tb, &k, &v],
        unit(),
        may_compact(call(
            "zb_dict_get_default",
            vec![hash_of(tb.e(), t), k.e(), nil()],
            any(),
        )),
    ));
    let ks = kept("ks", string());
    d.push(define(
        "zl_gc_before_insert_str",
        &[&tb, &ks, &v],
        unit(),
        may_compact(call(
            "zb_dict_get_default_str",
            vec![hash_of(tb.e(), t), ks.e(), nil()],
            any(),
        )),
    ));
    // At the end of the program, as `lua_close`: every object still
    // marked for finalization is finalized.
    d.push(define(
        "zl_gc_at_exit",
        &[],
        unit(),
        vec![
            when(
                ne(call("zl_gc_close", vec![], i64()), int(0)),
                vec![expr(call("zl_gc_finalize", vec![], unit()))],
            ),
            ret_void(),
        ],
    ));

    // `collectgarbage(opt, arg)`. Every option fails (nil) while a
    // finalizer runs, as the reference refuses collections then.
    let opt = kept("opt", string());
    let is = |name: &str| call("zb_str_eq", vec![opt.e(), text(name)], boolean());
    let arg = local("arg", i64());
    let mut body = vec![op.decl(int(-1))];
    for name in GC_OPTIONS {
        body.push(when(is(name), vec![op.set(int(option(name)))]));
    }
    let chosen = |name: &str| eq(op.e(), int(option(name)));
    body.extend([
        when(
            lt(op.e(), int(0)),
            vec![
                lua_error(concat(vec![
                    text("bad argument #1 to 'collectgarbage' (invalid option '"),
                    opt.e(),
                    text("')"),
                ])),
                ret(nil()),
            ],
        ),
        arg.decl(int(0)),
        when(
            and(
                or(chosen("setpause"), chosen("setstepmul")),
                not(is_nil(x.e())),
            ),
            vec![arg.set(call(
                "zl_arg_int",
                vec![x.e(), bad_arg(2, "collectgarbage")],
                i64(),
            ))],
        ),
        r.decl(call("zl_gc", vec![op.e(), arg.e()], i64())),
        when(eq(r.e(), int(-1)), vec![ret(nil())]),
        // A step is a whole cycle, which ends one in incremental mode;
        // in generational mode the reference's step never reports the
        // end of a cycle.
        when(
            or(chosen("collect"), chosen("step")),
            vec![
                expr(call("zl_gc_finalize", vec![], unit())),
                when(chosen("step"), vec![ret(box_bool(ne(r.e(), int(0))))]),
                ret(box_i64(int(0))),
            ],
        ),
        when(
            chosen("count"),
            vec![ret(box_f64(div(cast(r.e(), f64()), float(1024.0))))],
        ),
        when(chosen("isrunning"), vec![ret(box_bool(ne(r.e(), int(0))))]),
        when(
            or(chosen("incremental"), chosen("generational")),
            vec![ret(box_str(if_expr(
                ne(r.e(), int(0)),
                text("generational"),
                text("incremental"),
            )))],
        ),
        ret(box_i64(r.e())),
    ]);
    d.push(define("zl_collectgarbage", &[&opt, &x], any(), body));

    // `warn(msg, ...)`: every argument a string or a number, the
    // pieces handed on as one message.
    let first = kept("first", string());
    let rest = borrowed("rest", anys.clone());
    let cat = local("cat", i64());
    d.push(define(
        "zl_warn",
        &[&first, &rest],
        unit(),
        vec![
            n.decl(len(rest.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    y.decl(at(rest.e(), i.e())),
                    cat.decl(category(y.e())),
                    when(
                        not(or(is_number_cat(&cat), is_cat(&cat, STR))),
                        vec![
                            lua_error(concat(vec![
                                bad_arg_at(add(i.e(), int(1)), "warn"),
                                text(" (string expected, got "),
                                arg_type_name(y.e()),
                                text(")"),
                            ])),
                            ret_void(),
                        ],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            expr(call(
                "zl_warn_piece",
                vec![first.e(), gt(n.e(), int(0))],
                unit(),
            )),
            i.set(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    expr(call(
                        "zl_warn_piece",
                        vec![
                            call("zl_tostring", vec![at(rest.e(), i.e())], string()),
                            lt(i.e(), sub(n.e(), int(1))),
                        ],
                        unit(),
                    )),
                    i.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
    ));
    d
}
