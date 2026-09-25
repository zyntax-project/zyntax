//! `string.find`, `match`, `gmatch` and `gsub` over the host's pattern
//! matcher, which keeps the captures of the last match for the
//! library to read back.

use super::*;

pub fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let s = kept("s", string());
    let pat = kept("pat", string());
    let repl = kept("repl", any());
    let plain = kept("plain", any());
    let whole = local("whole", boolean());
    let init = local("init", i64());
    let max_n = local("max_n", i64());
    let i = local("i", i64());
    let n = local("n", i64());
    let k = local("k", i64());
    let e = local("e", i64());
    let src = local("src", i64());
    let last = local("last", i64());
    let level = local("level", i64());
    let kind = local("kind", i64());
    let anchor = local("anchor", boolean());
    let caps = local("caps", anys.clone());
    let out = local("out", anys.clone());
    let v = local("v", any());
    let env = kept("env", anys.clone());
    let packed = kept("packed", any());
    let done = local("done", boolean());
    let is_func = |x: Expr| {
        and(
            ne(x.clone(), nil()),
            eq(tag_of(x), int(zyntax_builtins::FUNC_TAG)),
        )
    };
    let is_number_cat_of = |x: Expr| {
        let c = category(x);
        or(
            or(eq(c.clone(), int(INT)), eq(c.clone(), int(UINT))),
            eq(c, int(FLOAT)),
        )
    };
    let mut d = Vec::new();

    for (name, params, ret_ty, symbol) in [
        (
            "zl_pat_specials",
            vec![("pat", string())],
            boolean(),
            "$Lua$pat_specials",
        ),
        (
            "zl_pat_find",
            vec![("s", string()), ("pat", string()), ("init", i64())],
            i64(),
            "$Lua$pat_find",
        ),
        (
            "zl_pat_match_at",
            vec![("s", string()), ("pat", string()), ("pos", i64())],
            i64(),
            "$Lua$pat_match_at",
        ),
        ("zl_pat_end", vec![], i64(), "$Lua$pat_end"),
        ("zl_pat_level", vec![], i64(), "$Lua$pat_level"),
        (
            "zl_pat_cap_kind",
            vec![("i", i64())],
            i64(),
            "$Lua$pat_cap_kind",
        ),
        (
            "zl_pat_cap_str",
            vec![("s", string()), ("i", i64())],
            string(),
            "$Lua$pat_cap_str",
        ),
        (
            "zl_pat_cap_pos",
            vec![("i", i64())],
            i64(),
            "$Lua$pat_cap_pos",
        ),
        ("zl_pat_error", vec![], string(), "$Lua$pat_error"),
        ("zl_buf_open", vec![], unit(), "$Lua$buf_open"),
        (
            "zl_buf_push",
            vec![("s", string())],
            unit(),
            "$Lua$buf_push",
        ),
        (
            "zl_buf_push_range",
            vec![("s", string()), ("a", i64()), ("b", i64())],
            unit(),
            "$Lua$buf_push_range",
        ),
        (
            "zl_buf_expand",
            vec![("s", string()), ("repl", string())],
            i64(),
            "$Lua$buf_expand",
        ),
        ("zl_buf_close", vec![], string(), "$Lua$buf_close"),
        (
            "zl_buf_replace_dots",
            vec![("s", string())],
            string(),
            "$Lua$replace_dots",
        ),
    ] {
        let params: Vec<(&str, Type)> = params.into_iter().collect();
        d.push(extern_fn(name, &params, ret_ty, Some(symbol)));
    }

    // The error the matcher held, raised.
    let pattern_error = || lua_error(call("zl_pat_error", vec![], string()));

    // Lua's string positions for a search: 1-based, negative from the
    // end, as a 0-based offset.
    d.push(define(
        "zl_pat_position",
        &[&init, &n],
        i64(),
        vec![
            when(gt(init.e(), int(0)), vec![ret(sub(init.e(), int(1)))]),
            when(
                or(eq(init.e(), int(0)), lt(init.e(), neg(n.e()))),
                vec![ret(int(0))],
            ),
            ret(add(n.e(), init.e())),
        ],
    ));

    // The captures of the last match as values: the whole match when
    // the pattern has none and `whole` asks for it.
    d.push(define(
        "zl_pat_captures",
        &[&s, &whole],
        anys.clone(),
        vec![
            level.decl(call("zl_pat_level", vec![], i64())),
            n.decl(if_expr(
                and(eq(level.e(), int(0)), whole.e()),
                int(1),
                level.e(),
            )),
            out.decl(list(vec![], anys.clone())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    kind.decl(call("zl_pat_cap_kind", vec![i.e()], i64())),
                    when(lt(kind.e(), int(0)), vec![pattern_error(), ret(out.e())]),
                    if_(
                        eq(kind.e(), int(1)),
                        vec![push(
                            out.e(),
                            box_i64(call("zl_pat_cap_pos", vec![i.e()], i64())),
                        )],
                        vec![push(
                            out.e(),
                            box_str(call("zl_pat_cap_str", vec![s.e(), i.e()], string())),
                        )],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            ret(out.e()),
        ],
    ));

    // `string.find(s, pattern, init, plain)`: the match's bounds and
    // its captures, or nil.
    d.push(define(
        "zl_string_find",
        &[&s, &pat, &init, &plain],
        any(),
        vec![
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            i.decl(call("zl_pat_position", vec![init.e(), n.e()], i64())),
            when(gt(i.e(), n.e()), vec![ret(nil())]),
            when(
                or(
                    call("zl_truthy", vec![plain.e()], boolean()),
                    not(call("zl_pat_specials", vec![pat.e()], boolean())),
                ),
                vec![
                    k.decl(call(
                        "zl_find_plain",
                        vec![s.e(), pat.e(), add(i.e(), int(1))],
                        i64(),
                    )),
                    when(eq(k.e(), int(0)), vec![ret(nil())]),
                    ret(call(
                        "zb_box_tuple",
                        vec![list(
                            vec![
                                box_i64(k.e()),
                                box_i64(sub(
                                    add(k.e(), call("zb_str_len", vec![pat.e()], i64())),
                                    int(1),
                                )),
                            ],
                            anys.clone(),
                        )],
                        any(),
                    )),
                ],
            ),
            k.decl(call("zl_pat_find", vec![s.e(), pat.e(), i.e()], i64())),
            when(eq(k.e(), int(-2)), vec![pattern_error(), ret(nil())]),
            when(lt(k.e(), int(0)), vec![ret(nil())]),
            out.decl(list(
                vec![
                    box_i64(add(k.e(), int(1))),
                    box_i64(call("zl_pat_end", vec![], i64())),
                ],
                anys.clone(),
            )),
            expr(call(
                "zb_list_extend_any",
                vec![
                    out.e(),
                    call("zl_pat_captures", vec![s.e(), bool(false)], anys.clone()),
                ],
                unit(),
            )),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));

    // `string.match(s, pattern, init)`: the captures, or the match.
    d.push(define(
        "zl_string_match",
        &[&s, &pat, &init],
        any(),
        vec![
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            i.decl(call("zl_pat_position", vec![init.e(), n.e()], i64())),
            when(gt(i.e(), n.e()), vec![ret(nil())]),
            k.decl(call("zl_pat_find", vec![s.e(), pat.e(), i.e()], i64())),
            when(eq(k.e(), int(-2)), vec![pattern_error(), ret(nil())]),
            when(lt(k.e(), int(0)), vec![ret(nil())]),
            ret(call(
                "zl_pack",
                vec![call(
                    "zl_pat_captures",
                    vec![s.e(), bool(true)],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));

    // `string.gmatch(s, pattern, init)`: an iterator over the matches.
    // Its record holds the subject, the pattern, the next position and
    // the end of the last match, so an empty match is not repeated
    // where the last one ended.
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    d.push(define(
        "zl_gmatch_code",
        &[&env, &packed],
        any(),
        vec![
            s.decl(get_str(at(env.e(), int(2)))),
            pat.decl(get_str(at(env.e(), int(3)))),
            src.decl(get_i64(at(env.e(), int(4)))),
            last.decl(get_i64(at(env.e(), int(5)))),
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            while_(
                le(src.e(), n.e()),
                vec![
                    e.decl(call(
                        "zl_pat_match_at",
                        vec![s.e(), pat.e(), src.e()],
                        i64(),
                    )),
                    when(eq(e.e(), int(-2)), vec![pattern_error(), ret(nil())]),
                    when(
                        and(ge(e.e(), int(0)), ne(e.e(), last.e())),
                        vec![
                            set_idx(env.e(), int(4), box_i64(e.e())),
                            set_idx(env.e(), int(5), box_i64(e.e())),
                            ret(call(
                                "zl_pack",
                                vec![call(
                                    "zl_pat_captures",
                                    vec![s.e(), bool(true)],
                                    anys.clone(),
                                )],
                                any(),
                            )),
                        ],
                    ),
                    src.add_assign(int(1)),
                ],
            ),
            set_idx(env.e(), int(4), box_i64(add(n.e(), int(1)))),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_string_gmatch",
        &[&s, &pat, &init],
        any(),
        vec![
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            i.decl(call("zl_pat_position", vec![init.e(), n.e()], i64())),
            when(gt(i.e(), n.e()), vec![i.set(add(n.e(), int(1)))]),
            ret(call(
                "zb_func_new",
                vec![
                    code_of("zl_gmatch_code"),
                    int(zyntax_builtins::functions::VARIADIC_ARITY),
                    list(
                        vec![
                            box_str(s.e()),
                            box_str(pat.e()),
                            box_i64(i.e()),
                            box_i64(int(-1)),
                        ],
                        anys.clone(),
                    ),
                ],
                any(),
            )),
        ],
    ));

    // One replacement of `gsub` into the open buffer: the string
    // expanded, or the table's or function's value for the match; a
    // false or nil value keeps the original text.
    let value_of = |v: &Local| {
        vec![
            when(
                not(call("zl_truthy", vec![v.e()], boolean())),
                vec![
                    expr(call(
                        "zl_buf_push_range",
                        vec![s.e(), src.e(), e.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                eq(category(v.e()), int(STR)),
                vec![
                    expr(call("zl_buf_push", vec![get_str(v.e())], unit())),
                    ret_void(),
                ],
            ),
            when(
                is_number_cat_of(v.e()),
                vec![
                    expr(call(
                        "zl_buf_push",
                        vec![call("zl_number_str", vec![v.e()], string())],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            lua_error(concat(vec![
                text("invalid replacement value (a "),
                type_name(v.e()),
                text(")"),
            ])),
            ret_void(),
        ]
    };
    d.push(define(
        "zl_gsub_add",
        &[&s, &repl, &src, &e],
        unit(),
        vec![
            when(
                eq(category(repl.e()), int(STR)),
                vec![
                    when(
                        eq(
                            call("zl_buf_expand", vec![s.e(), get_str(repl.e())], i64()),
                            int(-2),
                        ),
                        vec![pattern_error()],
                    ),
                    ret_void(),
                ],
            ),
            when(
                is_number_cat_of(repl.e()),
                vec![
                    when(
                        eq(
                            call(
                                "zl_buf_expand",
                                vec![s.e(), call("zl_number_str", vec![repl.e()], string())],
                                i64(),
                            ),
                            int(-2),
                        ),
                        vec![pattern_error()],
                    ),
                    ret_void(),
                ],
            ),
            caps.decl(call(
                "zl_pat_captures",
                vec![s.e(), bool(true)],
                anys.clone(),
            )),
            if_(
                is_table(repl.e()),
                vec![v.decl(call(
                    "zl_index",
                    vec![repl.e(), at(caps.e(), int(0))],
                    any(),
                ))],
                vec![v.decl(call(
                    "zl_first",
                    vec![call("zl_call_packed", vec![repl.e(), caps.e()], any())],
                    any(),
                ))],
            ),
            when(not(is_nil(pending())), vec![ret_void()]),
        ]
        .into_iter()
        .chain(value_of(&v))
        .collect(),
    ));

    // `string.gsub(s, pattern, repl, n)`: the subject with up to `n`
    // matches replaced, and how many were.
    d.push(define(
        "zl_string_gsub",
        &[&s, &pat, &repl, &max_n],
        any(),
        vec![
            when(
                not(or(
                    or(eq(category(repl.e()), int(STR)), is_number_cat_of(repl.e())),
                    or(is_table(repl.e()), is_func(repl.e())),
                )),
                vec![lua_error(concat(vec![
                    text("bad argument #3 to 'gsub' (string/function/table expected, got "),
                    arg_type_name(repl.e()),
                    text(")"),
                ]))],
            ),
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            anchor.decl(eq(
                call("zl_byte_at", vec![pat.e(), int(1)], i64()),
                int(94),
            )),
            when(
                anchor.e(),
                vec![pat.set(call(
                    "zl_string_sub",
                    vec![pat.e(), int(2), int(-1)],
                    string(),
                ))],
            ),
            expr(call("zl_buf_open", vec![], unit())),
            src.decl(int(0)),
            last.decl(int(-1)),
            k.decl(int(0)),
            done.decl(bool(false)),
            while_(
                and(lt(k.e(), max_n.e()), not(done.e())),
                vec![
                    e.decl(call(
                        "zl_pat_match_at",
                        vec![s.e(), pat.e(), src.e()],
                        i64(),
                    )),
                    when(
                        eq(e.e(), int(-2)),
                        vec![
                            expr(call("zl_buf_close", vec![], string())),
                            pattern_error(),
                            ret(nil()),
                        ],
                    ),
                    if_(
                        and(ge(e.e(), int(0)), ne(e.e(), last.e())),
                        vec![
                            k.add_assign(int(1)),
                            expr(call(
                                "zl_gsub_add",
                                vec![s.e(), repl.e(), src.e(), e.e()],
                                unit(),
                            )),
                            when(
                                not(is_nil(pending())),
                                vec![expr(call("zl_buf_close", vec![], string())), ret(nil())],
                            ),
                            src.set(e.e()),
                            last.set(e.e()),
                        ],
                        vec![if_(
                            lt(src.e(), n.e()),
                            vec![
                                expr(call(
                                    "zl_buf_push_range",
                                    vec![s.e(), src.e(), add(src.e(), int(1))],
                                    unit(),
                                )),
                                src.add_assign(int(1)),
                            ],
                            vec![done.set(bool(true))],
                        )],
                    ),
                    when(anchor.e(), vec![done.set(bool(true))]),
                ],
            ),
            expr(call(
                "zl_buf_push_range",
                vec![s.e(), src.e(), n.e()],
                unit(),
            )),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        box_str(call("zl_buf_close", vec![], string())),
                        box_i64(k.e()),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    d
}
