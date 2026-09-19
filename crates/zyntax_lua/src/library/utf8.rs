//! The `utf8` library over the host's decoder, which speaks Lua's own
//! UTF-8: sequences up to six bytes, strict unless asked otherwise.

use super::*;

pub fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let s = kept("s", string());
    let args = kept("args", anys.clone());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());
    let k = local("k", i64());
    let r = local("r", i64());
    let code = local("code", i64());
    let jv = kept("jv", any());
    let iv = kept("iv", any());
    let lax = kept("lax", any());
    let acc = local("acc", string());
    let out = local("out", anys.clone());
    let env = borrowed("env", anys.clone());
    let a0 = local("a0", any());
    let a1 = local("a1", any());
    let mut d = Vec::new();

    for (name, params, ret_ty, symbol) in [
        (
            "zl_utf8_char_of",
            vec![("code", i64())],
            string(),
            "$Lua$utf8_char",
        ),
        (
            "zl_utf8_len_raw",
            vec![
                ("s", string()),
                ("i", i64()),
                ("j", i64()),
                ("lax", boolean()),
            ],
            i64(),
            "$Lua$utf8_len",
        ),
        (
            "zl_utf8_offset_raw",
            vec![
                ("s", string()),
                ("n", i64()),
                ("i", i64()),
                ("has_i", boolean()),
            ],
            i64(),
            "$Lua$utf8_offset",
        ),
        (
            "zl_utf8_codepoint_raw",
            vec![
                ("s", string()),
                ("i", i64()),
                ("j", i64()),
                ("has_j", boolean()),
                ("lax", boolean()),
            ],
            i64(),
            "$Lua$utf8_codepoint",
        ),
        (
            "zl_utf8_code_at",
            vec![("k", i64())],
            i64(),
            "$Lua$utf8_code_at",
        ),
        (
            "zl_utf8_next",
            vec![("s", string()), ("n", i64()), ("lax", boolean())],
            i64(),
            "$Lua$utf8_next",
        ),
    ] {
        let params: Vec<(&str, Type)> = params.into_iter().collect();
        d.push(extern_fn(name, &params, ret_ty, Some(symbol)));
    }
    let truthy = |x: Expr| call("zl_truthy", vec![x], boolean());

    // `utf8.char(...)`: each code's sequence, concatenated.
    d.push(define(
        "zl_utf8_char",
        &[&args],
        string(),
        vec![
            acc.decl(text("")),
            i.decl(int(0)),
            while_(
                lt(i.e(), len(args.e())),
                vec![
                    code.decl(call(
                        "zl_arg_int",
                        vec![
                            at(args.e(), i.e()),
                            concat(vec![
                                text("bad argument #"),
                                call("zb_str_of_int", vec![add(i.e(), int(1))], string()),
                                text(" to 'char'"),
                            ]),
                        ],
                        i64(),
                    )),
                    when(
                        or(lt(code.e(), int(0)), gt(code.e(), int(0x7FFF_FFFF))),
                        vec![lua_error(concat(vec![
                            text("bad argument #"),
                            call("zb_str_of_int", vec![add(i.e(), int(1))], string()),
                            text(" to 'char' (value out of range)"),
                        ]))],
                    ),
                    acc.set(add(
                        acc.e(),
                        call("zl_utf8_char_of", vec![code.e()], string()),
                    )),
                    i.add_assign(int(1)),
                ],
            ),
            ret(acc.e()),
        ],
    ));

    // `utf8.len(s, i, j, lax)`: the count, or nil and where decoding
    // failed.
    d.push(define(
        "zl_utf8_len",
        &[&s, &i, &j, &lax],
        any(),
        vec![
            r.decl(call(
                "zl_utf8_len_raw",
                vec![s.e(), i.e(), j.e(), truthy(lax.e())],
                i64(),
            )),
            when(
                eq(r.e(), int(-1)),
                vec![lua_error(text(
                    "bad argument #2 to 'len' (initial position out of bounds)",
                ))],
            ),
            when(
                eq(r.e(), int(-2)),
                vec![lua_error(text(
                    "bad argument #3 to 'len' (final position out of bounds)",
                ))],
            ),
            when(
                lt(r.e(), int(0)),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![nil(), box_i64(neg(add(r.e(), int(2))))],
                        anys.clone(),
                    )],
                    any(),
                ))],
            ),
            ret(box_i64(r.e())),
        ],
    ));

    // `utf8.offset(s, n, i)`: the byte position of the `n`th character
    // from `i`, or nil.
    d.push(define(
        "zl_utf8_offset",
        &[&s, &n, &iv],
        any(),
        vec![
            i.decl(int(0)),
            when(
                not(is_nil(iv.e())),
                vec![i.set(call(
                    "zl_arg_int",
                    vec![iv.e(), bad_arg(3, "offset")],
                    i64(),
                ))],
            ),
            r.decl(call(
                "zl_utf8_offset_raw",
                vec![s.e(), n.e(), i.e(), not(is_nil(iv.e()))],
                i64(),
            )),
            when(
                eq(r.e(), int(-1)),
                vec![lua_error(text(
                    "bad argument #3 to 'offset' (position out of bounds)",
                ))],
            ),
            when(
                eq(r.e(), int(-2)),
                vec![lua_error(text("initial position is a continuation byte"))],
            ),
            when(eq(r.e(), int(0)), vec![ret(nil())]),
            ret(box_i64(r.e())),
        ],
    ));

    // `utf8.codepoint(s, i, j, lax)`: the codes of the characters that
    // start between `i` and `j`.
    d.push(define(
        "zl_utf8_codepoint",
        &[&s, &i, &jv, &lax],
        any(),
        vec![
            j.decl(int(0)),
            when(
                not(is_nil(jv.e())),
                vec![j.set(call(
                    "zl_arg_int",
                    vec![jv.e(), bad_arg(3, "codepoint")],
                    i64(),
                ))],
            ),
            r.decl(call(
                "zl_utf8_codepoint_raw",
                vec![s.e(), i.e(), j.e(), not(is_nil(jv.e())), truthy(lax.e())],
                i64(),
            )),
            when(
                eq(r.e(), int(-1)),
                vec![lua_error(text(
                    "bad argument #2 to 'codepoint' (out of bounds)",
                ))],
            ),
            when(
                eq(r.e(), int(-2)),
                vec![lua_error(text(
                    "bad argument #3 to 'codepoint' (out of bounds)",
                ))],
            ),
            when(
                eq(r.e(), int(-3)),
                vec![lua_error(text("invalid UTF-8 code"))],
            ),
            out.decl(list(vec![], anys.clone())),
            k.decl(int(0)),
            while_(
                lt(k.e(), r.e()),
                vec![
                    push(
                        out.e(),
                        box_i64(call("zl_utf8_code_at", vec![k.e()], i64())),
                    ),
                    k.add_assign(int(1)),
                ],
            ),
            ret(call("zb_box_tuple", vec![out.e()], any())),
        ],
    ));

    // `utf8.codes(s, lax)`: an iterator over positions and codes.
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    for (name, lax_flag) in [
        ("zl_utf8_codes_code", false),
        ("zl_utf8_codes_lax_code", true),
    ] {
        d.push(define(
            name,
            &[&env, &a0, &a1],
            any(),
            vec![
                s.decl(call(
                    "zl_arg_str",
                    vec![a0.e(), bad_arg(1, "for iterator")],
                    string(),
                )),
                n.decl(call(
                    "zl_arg_int",
                    vec![a1.e(), bad_arg(2, "for iterator")],
                    i64(),
                )),
                r.decl(call(
                    "zl_utf8_next",
                    vec![s.e(), n.e(), bool(lax_flag)],
                    i64(),
                )),
                when(
                    eq(r.e(), int(-1)),
                    vec![lua_error(text("invalid UTF-8 code"))],
                ),
                // The end: no values at all.
                when(eq(r.e(), int(0)), vec![ret(call("zl_none", vec![], any()))]),
                ret(call(
                    "zb_box_tuple",
                    vec![list(
                        vec![
                            box_i64(r.e()),
                            box_i64(call("zl_utf8_code_at", vec![int(0)], i64())),
                        ],
                        anys.clone(),
                    )],
                    any(),
                )),
            ],
        ));
    }
    let f = local("f", any());
    d.push(define(
        "zl_utf8_codes",
        &[&s, &lax],
        any(),
        vec![
            // A string starting inside a character is refused at once.
            when(
                and(
                    gt(call("zb_str_len", vec![s.e()], i64()), int(0)),
                    eq(
                        bitand(call("zl_byte_at", vec![s.e(), int(1)], i64()), int(0xC0)),
                        int(0x80),
                    ),
                ),
                vec![lua_error(text(
                    "bad argument #1 to 'codes' (invalid UTF-8 code)",
                ))],
            ),
            f.decl(call(
                "zl_func_of",
                vec![code_of("zl_utf8_codes_code"), int(2)],
                any(),
            )),
            when(
                truthy(lax.e()),
                vec![f.set(call(
                    "zl_func_of",
                    vec![code_of("zl_utf8_codes_lax_code"), int(2)],
                    any(),
                ))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![f.e(), box_str(s.e()), box_i64(int(0))],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    d
}
