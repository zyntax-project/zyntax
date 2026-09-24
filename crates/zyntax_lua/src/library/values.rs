//! Lua's operators on dynamic values: arithmetic with the integer and
//! float subtypes and string coercion, comparison, concatenation,
//! length, truth, and the conversions `tostring` and `tonumber`. Each
//! falls back to the metamethod of the event it implements.

use super::*;

/// The operator codes `zl_arith` takes, in the order of [`EVENTS`].
pub const OP_ADD: i64 = 0;
pub const OP_SUB: i64 = 1;
pub const OP_MUL: i64 = 2;
pub const OP_DIV: i64 = 3;
pub const OP_MOD: i64 = 4;
pub const OP_POW: i64 = 5;
pub const OP_IDIV: i64 = 6;
pub const OP_BAND: i64 = 7;
pub const OP_BOR: i64 = 8;
pub const OP_BXOR: i64 = 9;
pub const OP_SHL: i64 = 10;
pub const OP_SHR: i64 = 11;
pub const OP_CONCAT: i64 = 12;
pub const OP_UNM: i64 = 13;
pub const OP_BNOT: i64 = 14;

/// The metamethod each operator resolves to.
pub const EVENTS: [&str; 15] = [
    "__add", "__sub", "__mul", "__div", "__mod", "__pow", "__idiv", "__band", "__bor", "__bxor",
    "__shl", "__shr", "__concat", "__unm", "__bnot",
];

pub(super) fn declarations(_policy: &zyntax_builtins::Policy, t: &Types) -> Vec<Decl> {
    let x = kept("x", any());
    let a = kept("a", any());
    let b = kept("b", any());
    let op = local("op", i64());
    let ca = local("ca", i64());
    let cb = local("cb", i64());
    let cat = local("cat", i64());
    let ia = local("ia", i64());
    let ib = local("ib", i64());
    let fa = local("fa", f64());
    let fb = local("fb", f64());
    let f = local("f", f64());
    let i = local("i", i64());
    let n = local("n", i64());
    let s = kept("s", string());
    let h = kept("h", any());
    let r = kept("r", any());
    let na = kept("na", any());
    let nb = kept("nb", any());
    let mut d = Vec::new();
    let _ = &nb;

    // ─── the host's number primitives ───────────────────────────
    d.push(extern_fn(
        "zl_float_str",
        &[("x", f64())],
        string(),
        Some("$Lua$float_str"),
    ));
    d.push(extern_fn(
        "zl_number_kind",
        &[("s", string())],
        i64(),
        Some("$Lua$number_kind"),
    ));
    d.push(extern_fn(
        "zl_parse_int",
        &[("s", string())],
        i64(),
        Some("$Lua$parse_int"),
    ));
    d.push(extern_fn(
        "zl_parse_float",
        &[("s", string())],
        f64(),
        Some("$Lua$parse_float"),
    ));
    d.push(extern_fn(
        "zl_float_is_int",
        &[("x", f64())],
        boolean(),
        Some("$Lua$float_is_int"),
    ));

    // ─── truth ──────────────────────────────────────────────────
    // Only nil and false are false.
    d.push(define(
        "zl_truthy",
        &[&x],
        boolean(),
        vec![
            when(is_nil(x.e()), vec![ret(bool(false))]),
            when(eq(category(x.e()), int(BOOL)), vec![ret(get_bool(x.e()))]),
            ret(bool(true)),
        ],
    ));

    // ─── conversions ────────────────────────────────────────────
    // A number's text: integers in decimal, floats as `%.14g`.
    d.push(define(
        "zl_number_str",
        &[&x],
        string(),
        vec![
            cat.decl(category(x.e())),
            when(
                is_int_cat(&cat),
                vec![ret(call("zb_str_of_int", vec![get_i64(x.e())], string()))],
            ),
            ret(call("zl_float_str", vec![get_f64(x.e())], string())),
        ],
    ));
    // The text of a function value: its code address.
    let func_str = |x: Expr| {
        add(
            text("function: 0x"),
            call(
                "zb_str_of_int_radix",
                vec![call("zb_unbox_instance_raw", vec![x], i64()), int32(16)],
                string(),
            ),
        )
    };
    let is_func = |x: Expr| eq(tag_of(x), int(zyntax_builtins::FUNC_TAG));
    let is_code = |x: Expr| eq(tag_of(x), int(zyntax_builtins::CODE_TAG));
    d.push(define(
        "zl_tostring",
        &[&x],
        string(),
        vec![
            when(is_nil(x.e()), vec![ret(text("nil"))]),
            cat.decl(category(x.e())),
            when(is_cat(&cat, STR), vec![ret(get_str(x.e()))]),
            when(
                is_number_cat(&cat),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            when(
                is_cat(&cat, BOOL),
                vec![ret(if_expr(get_bool(x.e()), text("true"), text("false")))],
            ),
            when(
                is_table(x.e()),
                vec![
                    h.decl(call("zl_meta_of", vec![x.e(), text("__tostring")], any())),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            r.decl(call(
                                "zl_first",
                                vec![call("zl_call_1", vec![h.e(), x.e()], any())],
                                any(),
                            )),
                            when(
                                and(
                                    not(is_nil(r.e())),
                                    or(
                                        eq(category(r.e()), int(INT)),
                                        or(
                                            eq(category(r.e()), int(UINT)),
                                            eq(category(r.e()), int(FLOAT)),
                                        ),
                                    ),
                                ),
                                vec![ret(call("zl_number_str", vec![r.e()], string()))],
                            ),
                            when(
                                or(is_nil(r.e()), ne(category(r.e()), int(STR))),
                                vec![lua_error(text("'__tostring' must return a string"))],
                            ),
                            ret(get_str(r.e())),
                        ],
                    ),
                    h.set(call("zl_meta_of", vec![x.e(), text("__name")], any())),
                    when(
                        eq(category(h.e()), int(STR)),
                        vec![ret(add(
                            add(get_str(h.e()), text(": 0x")),
                            call(
                                "zb_str_of_int_radix",
                                vec![call("zb_unbox_instance_raw", vec![x.e()], i64()), int32(16)],
                                string(),
                            ),
                        ))],
                    ),
                    ret(call("zb_hook_instance_str", vec![x.e()], string())),
                ],
            ),
            when(
                or(is_thread(x.e()), is_file(x.e())),
                vec![ret(call("zb_hook_instance_str", vec![x.e()], string()))],
            ),
            when(
                or(is_func(x.e()), is_code(x.e())),
                vec![ret(func_str(x.e()))],
            ),
            ret(call("zb_any_str", vec![x.e()], string())),
        ],
    ));
    // `tonumber(x)`: a number as it is, a numeral's value, else nil.
    d.push(define(
        "zl_str_to_number",
        &[&s],
        any(),
        vec![
            n.decl(call("zl_number_kind", vec![s.e()], i64())),
            when(
                eq(n.e(), int(1)),
                vec![ret(box_i64(call("zl_parse_int", vec![s.e()], i64())))],
            ),
            when(
                eq(n.e(), int(2)),
                vec![ret(box_f64(call("zl_parse_float", vec![s.e()], f64())))],
            ),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_tonumber",
        &[&x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            cat.decl(category(x.e())),
            when(is_number_cat(&cat), vec![ret(x.e())]),
            when(
                is_cat(&cat, STR),
                vec![ret(call("zl_str_to_number", vec![get_str(x.e())], any()))],
            ),
            ret(nil()),
        ],
    ));
    // `tonumber(s, base)` for an integer numeral in another base.
    let base = local("base", i64());
    let digit = local("digit", i64());
    let acc = local("acc", i64());
    let neg = local("neg", boolean());
    let code = local("code", i64());
    let text_in = kept("text", string());
    d.push(define(
        "zl_tonumber_base",
        &[&text_in, &base],
        any(),
        vec![
            when(
                or(lt(base.e(), int(2)), gt(base.e(), int(36))),
                vec![lua_error(text(
                    "bad argument #2 to 'tonumber' (base out of range)",
                ))],
            ),
            s.decl(call(
                "zb_str_lower",
                vec![call("zb_str_strip", vec![text_in.e()], string())],
                string(),
            )),
            n.decl(call("zb_str_len", vec![s.e()], i64())),
            when(eq(n.e(), int(0)), vec![ret(nil())]),
            i.decl(int(0)),
            neg.decl(bool(false)),
            code.decl(cast(
                call("zb_str_code_at", vec![s.e(), int(0)], i32()),
                i64(),
            )),
            when(
                eq(code.e(), int(45)),
                vec![neg.set(bool(true)), i.set(int(1))],
            ),
            when(eq(code.e(), int(43)), vec![i.set(int(1))]),
            when(ge(i.e(), n.e()), vec![ret(nil())]),
            acc.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    code.set(cast(
                        call("zb_str_code_at", vec![s.e(), i.e()], i32()),
                        i64(),
                    )),
                    digit.decl(int(-1)),
                    when(
                        and(ge(code.e(), int(48)), le(code.e(), int(57))),
                        vec![digit.set(sub(code.e(), int(48)))],
                    ),
                    when(
                        and(ge(code.e(), int(97)), le(code.e(), int(122))),
                        vec![digit.set(add(sub(code.e(), int(97)), int(10)))],
                    ),
                    when(
                        or(lt(digit.e(), int(0)), ge(digit.e(), base.e())),
                        vec![ret(nil())],
                    ),
                    acc.set(add(mul(acc.e(), base.e()), digit.e())),
                    i.add_assign(int(1)),
                ],
            ),
            when(neg.e(), vec![acc.set(sub(int(0), acc.e()))]),
            ret(box_i64(acc.e())),
        ],
    ));

    // ─── arithmetic ─────────────────────────────────────────────
    // The number an arithmetic operand is: itself, or the numeral a
    // string holds; nil when it is neither.
    d.push(define(
        "zl_arith_operand",
        &[&x],
        any(),
        vec![
            when(is_nil(x.e()), vec![ret(nil())]),
            cat.decl(category(x.e())),
            when(is_number_cat(&cat), vec![ret(x.e())]),
            when(
                is_cat(&cat, STR),
                vec![ret(call("zl_str_to_number", vec![get_str(x.e())], any()))],
            ),
            ret(nil()),
        ],
    ));
    // The event's name by code, or the operation's without the
    // underscores; a chain of comparisons keeps it in typed AST.
    let name_chain = |op: Expr, skip: usize| {
        let mut e = text(&EVENTS[EVENTS.len() - 1][skip..]);
        for (i, name) in EVENTS.iter().enumerate().rev().skip(1) {
            e = if_expr(eq(op.clone(), int(i as i64)), text(&name[skip..]), e);
        }
        e
    };
    let event_name = |op: Expr| name_chain(op, 0);
    let operation_name = |op: Expr| name_chain(op, 2);
    let is_text_like = |x: Expr| {
        let c = category(x);
        or(
            eq(c.clone(), int(STR)),
            or(
                eq(c.clone(), int(INT)),
                or(eq(c.clone(), int(UINT)), eq(c, int(FLOAT))),
            ),
        )
    };
    // Whether a number has an integer value.
    let has_int = |x: Expr| {
        or(
            is_int_cat_of(x.clone()),
            call("zl_float_is_int", vec![get_f64(x)], boolean()),
        )
    };
    let is_str = |x: Expr| and(not(is_nil(x.clone())), eq(category(x), int(STR)));
    let is_number = |x: Expr| {
        and(
            not(is_nil(x.clone())),
            or(is_int_cat_of(x.clone()), eq(category(x), int(FLOAT))),
        )
    };
    // The metamethod for `op` on `a` or `b`, called; or the error,
    // about the first operand that is wrong: for `..` the first that is
    // not text, otherwise the first that is not a number. A string
    // whose arithmetic fails errs the way the string library's
    // metamethods do.
    let bad = local("bad", i64());
    d.push(define(
        "zl_arith_meta",
        &[&op, &a, &b],
        any(),
        vec![
            s.decl(event_name(op.e())),
            h.decl(call("zl_meta_of", vec![a.e(), s.e()], any())),
            when(
                is_nil(h.e()),
                vec![h.set(call("zl_meta_of", vec![b.e(), s.e()], any()))],
            ),
            when(
                not(is_nil(h.e())),
                vec![
                    metamethod_call_check(h.e(), operation_name(op.e())),
                    ret(call(
                        "zl_first",
                        vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                        any(),
                    )),
                ],
            ),
            when(
                eq(op.e(), int(OP_CONCAT)),
                vec![
                    bad.decl(if_expr(
                        and(not(is_nil(a.e())), is_text_like(a.e())),
                        int(OPERAND_RIGHT),
                        int(OPERAND_LEFT),
                    )),
                    x.decl(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
                    type_error(
                        concat(vec![
                            text("attempt to concatenate a "),
                            type_name(x.e()),
                            text(" value"),
                        ]),
                        bad.e(),
                    ),
                    ret(nil()),
                ],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            nb.decl(call("zl_arith_operand", vec![b.e()], any())),
            bad.decl(if_expr(
                is_nil(na.e()),
                int(OPERAND_LEFT),
                int(OPERAND_RIGHT),
            )),
            x.decl(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
            // A bitwise operand is a number by type: two numbers, one
            // without an integer value, or the first that is not one.
            when(
                or(
                    and(ge(op.e(), int(OP_BAND)), le(op.e(), int(OP_SHR))),
                    eq(op.e(), int(OP_BNOT)),
                ),
                vec![
                    when(
                        and(is_number(a.e()), is_number(b.e())),
                        vec![
                            bad.set(if_expr(
                                has_int(a.e()),
                                int(OPERAND_RIGHT),
                                int(OPERAND_LEFT),
                            )),
                            type_error(
                                text("number has no integer representation"),
                                bitor(bad.e(), int(VARINFO_INSIDE)),
                            ),
                            ret(nil()),
                        ],
                    ),
                    bad.set(if_expr(
                        is_number(a.e()),
                        int(OPERAND_RIGHT),
                        int(OPERAND_LEFT),
                    )),
                    x.set(if_expr(eq(bad.e(), int(OPERAND_LEFT)), a.e(), b.e())),
                    type_error(
                        concat(vec![
                            text("attempt to perform bitwise operation on a "),
                            type_name(x.e()),
                            text(" value"),
                        ]),
                        bad.e(),
                    ),
                    ret(nil()),
                ],
            ),
            when(
                or(is_str(a.e()), is_str(b.e())),
                vec![lua_error(concat(vec![
                    text("attempt to "),
                    operation_name(op.e()),
                    text(" a '"),
                    type_name(a.e()),
                    text("' with a '"),
                    type_name(b.e()),
                    text("'"),
                ]))],
            ),
            type_error(
                concat(vec![
                    text("attempt to perform arithmetic on a "),
                    type_name(x.e()),
                    text(" value"),
                ]),
                bad.e(),
            ),
            ret(nil()),
        ],
    ));
    // The integer a bitwise operand is, or -1 with `ok` false: an
    // integer, a float with an integral value, or a numeral string.
    // `out = the integer of x`, a number already; `scratch` holds the
    // float on the way. `fail` runs when it has no integer value.
    let to_int_stmts = |x: &Local, out: &Local, scratch: &Local, fail: Vec<Stmt>| {
        vec![if_(
            is_int_cat_of(x.e()),
            vec![out.set(get_i64(x.e()))],
            vec![
                scratch.set(get_f64(x.e())),
                when(
                    not(call("zl_float_is_int", vec![scratch.e()], boolean())),
                    fail,
                ),
                out.set(cast(scratch.e(), i64())),
            ],
        )]
    };
    fn is_int_cat_of(x: Expr) -> Expr {
        let c = category(x);
        or(eq(c.clone(), int(INT)), eq(c, int(UINT)))
    }
    // The initial value or step of a `for`, as the loop's number type;
    // `what` names which for the message.
    let what = kept("what", string());
    let for_number = |x: &Local, what: &Local| {
        vec![
            na.decl(call("zl_arith_operand", vec![x.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("bad 'for' "),
                    what.e(),
                    text(" (number expected, got "),
                    type_name(x.e()),
                    text(")"),
                ]))],
            ),
        ]
    };
    let mut st = for_number(&a, &what);
    st.extend([
        when(is_int_cat_of(na.e()), vec![ret(get_i64(na.e()))]),
        f.decl(get_f64(na.e())),
        when(
            not(call("zl_float_is_int", vec![f.e()], boolean())),
            vec![lua_error(concat(vec![
                text("'for' "),
                what.e(),
                text(" must be an integer"),
            ]))],
        ),
        ret(cast(f.e(), i64())),
    ]);
    d.push(define("zl_for_int", &[&a, &what], i64(), st));
    let mut st = for_number(&a, &what);
    st.extend([
        when(
            is_int_cat_of(na.e()),
            vec![ret(cast(get_i64(na.e()), f64()))],
        ),
        ret(get_f64(na.e())),
    ]);
    d.push(define("zl_for_float", &[&a, &what], f64(), st));
    // Whether a `for` whose start and step are dynamic values counts in
    // integers: both are integers, not numerals or floats.
    d.push(define(
        "zl_for_ints",
        &[&a, &b],
        boolean(),
        vec![
            when(or(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(false))]),
            ret(and(is_int_cat_of(a.e()), is_int_cat_of(b.e()))),
        ],
    ));
    // The integer limit of a `for` over integers with a float limit:
    // the last integer the loop may reach, so floored when the step
    // climbs and rounded up when it falls. Past the integers on the
    // step's own side it is clamped, which the loop never runs past;
    // on the other side the loop has no iteration, which the flag
    // says, since no integer limit could.
    let step = local("step", i64());
    let fl = local("fl", f64());
    d.push(define(
        "zl_for_limit_f",
        &[&fl, &step],
        i64(),
        vec![
            set_global(FOR_SKIP, bool(false)),
            f.decl(if_expr(
                lt(step.e(), int(0)),
                call("zl_ceil_f64", vec![fl.e()], f64()),
                call("floor", vec![fl.e()], f64()),
            )),
            when(
                ge(f.e(), float(9223372036854775808.0)),
                vec![
                    when(lt(step.e(), int(0)), vec![set_global(FOR_SKIP, bool(true))]),
                    ret(int(i64::MAX)),
                ],
            ),
            when(
                lt(f.e(), float(-9223372036854775808.0)),
                vec![
                    when(gt(step.e(), int(0)), vec![set_global(FOR_SKIP, bool(true))]),
                    ret(int(i64::MIN)),
                ],
            ),
            ret(cast(f.e(), i64())),
        ],
    ));
    // The same from a dynamic limit: a numeral read.
    d.push(define(
        "zl_for_limit",
        &[&a, &step],
        i64(),
        vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("bad 'for' limit (number expected, got "),
                    type_name(a.e()),
                    text(")"),
                ]))],
            ),
            when(
                is_int_cat_of(na.e()),
                vec![set_global(FOR_SKIP, bool(false)), ret(get_i64(na.e()))],
            ),
            ret(call(
                "zl_for_limit_f",
                vec![get_f64(na.e()), step.e()],
                i64(),
            )),
        ],
    ));
    // Typed helpers with Lua's rules, shared with the typed fast paths.
    // Division by -1 is a negation that wraps, never a trap.
    d.push(define(
        "zl_idiv_i64",
        &[&ia, &ib],
        i64(),
        vec![
            when(
                eq(ib.e(), int(0)),
                vec![lua_error(text("attempt to divide by zero"))],
            ),
            when(eq(ib.e(), int(-1)), vec![ret(sub(int(0), ia.e()))]),
            ret(call("zb_floordiv_i64", vec![ia.e(), ib.e()], i64())),
        ],
    ));
    d.push(define(
        "zl_mod_i64",
        &[&ia, &ib],
        i64(),
        vec![
            when(
                eq(ib.e(), int(0)),
                vec![lua_error(text("attempt to perform 'n%0'"))],
            ),
            when(eq(ib.e(), int(-1)), vec![ret(int(0))]),
            ret(call("zb_mod_i64", vec![ia.e(), ib.e()], i64())),
        ],
    ));
    d.push(define(
        "zl_idiv_f64",
        &[&fa, &fb],
        f64(),
        vec![ret(call("floor", vec![div(fa.e(), fb.e())], f64()))],
    ));
    // `a - floor(a/b)*b`, except that an infinite divisor keeps a finite
    // dividend of the other sign as itself plus the divisor's sign.
    d.push(define(
        "zl_mod_f64",
        &[&fa, &fb],
        f64(),
        vec![
            when(
                or(
                    eq(fb.e(), float(f64::INFINITY)),
                    eq(fb.e(), float(f64::NEG_INFINITY)),
                ),
                vec![
                    when(ne(fa.e(), fa.e()), vec![ret(fa.e())]),
                    when(
                        or(
                            eq(fa.e(), float(f64::INFINITY)),
                            eq(fa.e(), float(f64::NEG_INFINITY)),
                        ),
                        vec![ret(sub(fa.e(), fa.e()))],
                    ),
                    when(
                        or(
                            and(ge(fa.e(), float(0.0)), gt(fb.e(), float(0.0))),
                            and(le(fa.e(), float(0.0)), lt(fb.e(), float(0.0))),
                        ),
                        vec![ret(fa.e())],
                    ),
                    ret(fb.e()),
                ],
            ),
            ret(call("zb_mod_f64", vec![fa.e(), fb.e()], f64())),
        ],
    ));
    d.push(define(
        "zl_pow",
        &[&fa, &fb],
        f64(),
        vec![ret(call("pow", vec![fa.e(), fb.e()], f64()))],
    ));
    // Shifts: by 64 or more is zero, a negative count shifts the other
    // way, and a right shift is logical.
    d.push(define(
        "zl_shl_i64",
        &[&ia, &ib],
        i64(),
        vec![
            // A shift by 64 or more, either way, is zero; a negative
            // shift is the other direction.
            when(
                or(ge(ib.e(), int(64)), le(ib.e(), int(-64))),
                vec![ret(int(0))],
            ),
            when(
                lt(ib.e(), int(0)),
                vec![ret(call(
                    "zl_shr_i64",
                    vec![ia.e(), sub(int(0), ib.e())],
                    i64(),
                ))],
            ),
            ret(shl(ia.e(), ib.e())),
        ],
    ));
    // The shift right is logical: the bits the arithmetic shift fills
    // with the sign are masked off.
    d.push(define(
        "zl_shr_i64",
        &[&ia, &ib],
        i64(),
        vec![
            when(
                or(ge(ib.e(), int(64)), le(ib.e(), int(-64))),
                vec![ret(int(0))],
            ),
            when(
                lt(ib.e(), int(0)),
                vec![ret(call(
                    "zl_shl_i64",
                    vec![ia.e(), sub(int(0), ib.e())],
                    i64(),
                ))],
            ),
            when(eq(ib.e(), int(0)), vec![ret(ia.e())]),
            ret(bitand(
                shr(ia.e(), ib.e()),
                sub(shl(int(1), sub(int(64), ib.e())), int(1)),
            )),
        ],
    ));
    // The operator on two integers, boxed.
    let int_op = |op: &Local, a: Expr, b: Expr| -> Vec<Stmt> {
        let pick = |code: i64, v: Expr| when(eq(op.e(), int(code)), vec![ret(box_i64(v))]);
        vec![
            pick(OP_ADD, add(a.clone(), b.clone())),
            pick(OP_SUB, sub(a.clone(), b.clone())),
            pick(OP_MUL, mul(a.clone(), b.clone())),
            pick(
                OP_IDIV,
                call("zl_idiv_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(
                OP_MOD,
                call("zl_mod_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(OP_BAND, bitand(a.clone(), b.clone())),
            pick(OP_BOR, bitor(a.clone(), b.clone())),
            pick(OP_BXOR, bitxor(a.clone(), b.clone())),
            pick(
                OP_SHL,
                call("zl_shl_i64", vec![a.clone(), b.clone()], i64()),
            ),
            pick(OP_SHR, call("zl_shr_i64", vec![a, b], i64())),
        ]
    };
    let float_op = |op: &Local, a: Expr, b: Expr| -> Vec<Stmt> {
        let pick = |code: i64, v: Expr| when(eq(op.e(), int(code)), vec![ret(box_f64(v))]);
        vec![
            pick(OP_ADD, add(a.clone(), b.clone())),
            pick(OP_SUB, sub(a.clone(), b.clone())),
            pick(OP_MUL, mul(a.clone(), b.clone())),
            pick(OP_DIV, div(a.clone(), b.clone())),
            pick(OP_POW, call("zl_pow", vec![a.clone(), b.clone()], f64())),
            pick(
                OP_IDIV,
                call("zl_idiv_f64", vec![a.clone(), b.clone()], f64()),
            ),
            pick(OP_MOD, call("zl_mod_f64", vec![a, b], f64())),
        ]
    };
    let as_f64 = |x: Expr| {
        if_expr(
            is_int_cat_of(x.clone()),
            cast(get_i64(x.clone()), f64()),
            get_f64(x),
        )
    };
    // `a op b` on dynamic values. Tables go to their metamethods first;
    // otherwise both operands are numbers, integers when both are and
    // the operator keeps integers.
    let na = kept("na", any());
    let nb = kept("nb", any());
    d.push(define("zl_arith", &[&op, &a, &b], any(), {
        let mut st = vec![
            when(
                eq(op.e(), int(OP_CONCAT)),
                vec![ret(call("zl_concat", vec![a.e(), b.e()], any()))],
            ),
            when(
                or(is_table(a.e()), is_table(b.e())),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![op.e(), a.e(), b.e()],
                    any(),
                ))],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            nb.decl(call("zl_arith_operand", vec![b.e()], any())),
            when(
                or(is_nil(na.e()), is_nil(nb.e())),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![op.e(), a.e(), b.e()],
                    any(),
                ))],
            ),
        ];
        // Bitwise operators convert floats with integral values, never
        // strings: those are the string metatable's to handle.
        let meta = || ret(call("zl_arith_meta", vec![op.e(), a.e(), b.e()], any()));
        let mut bitwise = vec![when(
            or(eq(category(a.e()), int(STR)), eq(category(b.e()), int(STR))),
            vec![meta()],
        )];
        bitwise.extend(to_int_stmts(&na, &ia, &fa, vec![meta()]));
        bitwise.extend(to_int_stmts(&nb, &ib, &fb, vec![meta()]));
        bitwise.extend(int_op(&op, ia.e(), ib.e()));
        st.push(ia.decl(int(0)));
        st.push(ib.decl(int(0)));
        st.push(fa.decl(float(0.0)));
        st.push(fb.decl(float(0.0)));
        st.push(when(
            and(ge(op.e(), int(OP_BAND)), le(op.e(), int(OP_SHR))),
            bitwise,
        ));
        let mut ints = vec![ia.set(get_i64(na.e())), ib.set(get_i64(nb.e()))];
        ints.extend(int_op(&op, ia.e(), ib.e()));
        st.push(when(
            and(
                and(is_int_cat_of(na.e()), is_int_cat_of(nb.e())),
                and(ne(op.e(), int(OP_DIV)), ne(op.e(), int(OP_POW))),
            ),
            ints,
        ));
        st.push(fa.set(as_f64(na.e())));
        st.push(fb.set(as_f64(nb.e())));
        st.extend(float_op(&op, fa.e(), fb.e()));
        st.push(ret(nil()));
        st
    }));
    // Unary minus and bitwise not.
    d.push(define(
        "zl_unm",
        &[&a],
        any(),
        vec![
            when(
                is_table(a.e()),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![int(OP_UNM), a.e(), a.e()],
                    any(),
                ))],
            ),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![ret(call(
                    "zl_arith_meta",
                    vec![int(OP_UNM), a.e(), a.e()],
                    any(),
                ))],
            ),
            when(
                is_int_cat_of(na.e()),
                vec![ret(box_i64(sub(int(0), get_i64(na.e()))))],
            ),
            ret(box_f64(sub(float(0.0), get_f64(na.e())))),
        ],
    ));
    d.push(define("zl_bnot", &[&a], any(), {
        let meta = || {
            ret(call(
                "zl_arith_meta",
                vec![int(OP_BNOT), a.e(), a.e()],
                any(),
            ))
        };
        let mut st = vec![
            when(or(is_table(a.e()), is_str(a.e())), vec![meta()]),
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(is_nil(na.e()), vec![meta()]),
            ia.decl(int(0)),
            fa.decl(float(0.0)),
        ];
        st.extend(to_int_stmts(&na, &ia, &fa, vec![meta()]));
        st.push(ret(box_i64(bitxor(ia.e(), int(-1)))));
        st
    }));
    // The integer a dynamic value holds, for a typed bitwise operand.
    d.push(define("zl_toint", &[&a], i64(), {
        let mut st = vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("attempt to perform bitwise operation on a "),
                    type_name(a.e()),
                    text(" value"),
                ]))],
            ),
            ia.decl(int(0)),
            fa.decl(float(0.0)),
        ];
        st.extend(to_int_stmts(
            &na,
            &ia,
            &fa,
            vec![lua_error(text("number has no integer representation"))],
        ));
        st.push(ret(ia.e()));
        st
    }));
    // The float a dynamic value holds, for a typed float operand.
    d.push(define(
        "zl_tofloat",
        &[&a],
        f64(),
        vec![
            na.decl(call("zl_arith_operand", vec![a.e()], any())),
            when(
                is_nil(na.e()),
                vec![lua_error(concat(vec![
                    text("attempt to perform arithmetic on a "),
                    type_name(a.e()),
                    text(" value"),
                ]))],
            ),
            ret(as_f64(na.e())),
        ],
    ));

    // ─── comparison ─────────────────────────────────────────────
    // An integer against a float, exactly: the float is brought to an
    // integer when it has one in range, otherwise its side of the
    // range decides. An integer within 2^53 compares as a float.
    let fits_float = |i: Expr| and(ge(i.clone(), int(-(1i64 << 53))), le(i, int(1i64 << 53)));
    let in_range = |x: Expr| {
        and(
            ge(x.clone(), float(-9223372036854775808.0)),
            lt(x, float(9223372036854775808.0)),
        )
    };
    d.push(define(
        "zl_eq_if",
        &[&ia, &fa],
        boolean(),
        vec![
            // An integer a float holds exactly equals the float that is
            // it, and nothing else.
            when(
                fits_float(ia.e()),
                vec![ret(eq(cast(ia.e(), f64()), fa.e()))],
            ),
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                ne(call("floor", vec![fa.e()], f64()), fa.e()),
                vec![ret(bool(false))],
            ),
            when(not(in_range(fa.e())), vec![ret(bool(false))]),
            ret(eq(ia.e(), cast(fa.e(), i64()))),
        ],
    ));
    // `i < f`: with the float's ceiling; `i <= f`: with its floor.
    d.push(define(
        "zl_lt_if",
        &[&ia, &fa],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(lt(cast(ia.e(), f64()), fa.e()))],
            ),
            f.decl(call("zl_ceil_f64", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(lt(ia.e(), cast(f.e(), i64())))]),
            ret(gt(fa.e(), float(0.0))),
        ],
    ));
    d.push(define(
        "zl_le_if",
        &[&ia, &fa],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(le(cast(ia.e(), f64()), fa.e()))],
            ),
            f.decl(call("floor", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(le(ia.e(), cast(f.e(), i64())))]),
            ret(gt(fa.e(), float(0.0))),
        ],
    ));
    // `f < i`: with the float's floor; `f <= i`: with its ceiling.
    d.push(define(
        "zl_lt_fi",
        &[&fa, &ia],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(lt(fa.e(), cast(ia.e(), f64())))],
            ),
            f.decl(call("floor", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(lt(cast(f.e(), i64()), ia.e()))]),
            ret(lt(fa.e(), float(0.0))),
        ],
    ));
    d.push(define(
        "zl_le_fi",
        &[&fa, &ia],
        boolean(),
        vec![
            when(ne(fa.e(), fa.e()), vec![ret(bool(false))]),
            when(
                fits_float(ia.e()),
                vec![ret(le(fa.e(), cast(ia.e(), f64())))],
            ),
            f.decl(call("zl_ceil_f64", vec![fa.e()], f64())),
            when(in_range(f.e()), vec![ret(le(cast(f.e(), i64()), ia.e()))]),
            ret(lt(fa.e(), float(0.0))),
        ],
    ));
    let num_eq = |a: Expr, b: Expr| {
        if_expr(
            and(is_int_cat_of(a.clone()), is_int_cat_of(b.clone())),
            eq(get_i64(a.clone()), get_i64(b.clone())),
            if_expr(
                is_int_cat_of(a.clone()),
                call(
                    "zl_eq_if",
                    vec![get_i64(a.clone()), get_f64(b.clone())],
                    boolean(),
                ),
                if_expr(
                    is_int_cat_of(b.clone()),
                    call(
                        "zl_eq_if",
                        vec![get_i64(b.clone()), get_f64(a.clone())],
                        boolean(),
                    ),
                    eq(get_f64(a), get_f64(b)),
                ),
            ),
        )
    };
    d.push(define(
        "zl_eq",
        &[&a, &b],
        boolean(),
        vec![
            when(and(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(true))]),
            when(or(is_nil(a.e()), is_nil(b.e())), vec![ret(bool(false))]),
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            // Numbers compare by value first: a NaN is not itself,
            // whichever box holds it.
            when(
                and(is_number_cat(&ca), is_number_cat(&cb)),
                vec![ret(num_eq(a.e(), b.e()))],
            ),
            when(eq(a.e(), b.e()), vec![ret(bool(true))]),
            when(
                and(is_cat(&ca, STR), is_cat(&cb, STR)),
                vec![ret(str_eq(get_str(a.e()), get_str(b.e())))],
            ),
            when(
                and(is_cat(&ca, BOOL), is_cat(&cb, BOOL)),
                vec![ret(eq(get_bool(a.e()), get_bool(b.e())))],
            ),
            when(
                or(ne(ca.e(), int(CUSTOM)), ne(cb.e(), int(CUSTOM))),
                vec![ret(bool(false))],
            ),
            when(
                eq(
                    call("zb_unbox_instance_raw", vec![a.e()], i64()),
                    call("zb_unbox_instance_raw", vec![b.e()], i64()),
                ),
                vec![ret(bool(true))],
            ),
            when(
                and(is_table(a.e()), is_table(b.e())),
                vec![
                    h.decl(call("zl_meta_of", vec![a.e(), text("__eq")], any())),
                    when(
                        is_nil(h.e()),
                        vec![h.set(call("zl_meta_of", vec![b.e(), text("__eq")], any()))],
                    ),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            metamethod_call_check(h.e(), text("eq")),
                            ret(call(
                                "zl_truthy",
                                vec![call(
                                    "zl_first",
                                    vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                                    any(),
                                )],
                                boolean(),
                            )),
                        ],
                    ),
                ],
            ),
            ret(bool(false)),
        ],
    ));
    let compare_error = |a: &Local, b: &Local| {
        vec![
            if_(
                str_eq(type_name(a.e()), type_name(b.e())),
                vec![lua_error(concat(vec![
                    text("attempt to compare two "),
                    type_name(a.e()),
                    text(" values"),
                ]))],
                vec![lua_error(concat(vec![
                    text("attempt to compare "),
                    type_name(a.e()),
                    text(" with "),
                    type_name(b.e()),
                ]))],
            ),
            ret(bool(false)),
        ]
    };
    let order = |name: &str,
                 event: &str,
                 int_cmp: fn(Expr, Expr) -> Expr,
                 str_test: fn(Expr) -> Expr,
                 mixed: (&str, &str)| {
        let mut st = vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                and(is_number_cat(&ca), is_number_cat(&cb)),
                vec![
                    when(
                        and(is_int_cat_of(a.e()), is_int_cat_of(b.e())),
                        vec![ret(int_cmp(get_i64(a.e()), get_i64(b.e())))],
                    ),
                    when(
                        is_int_cat_of(a.e()),
                        vec![ret(call(
                            mixed.0,
                            vec![get_i64(a.e()), get_f64(b.e())],
                            boolean(),
                        ))],
                    ),
                    when(
                        is_int_cat_of(b.e()),
                        vec![ret(call(
                            mixed.1,
                            vec![get_f64(a.e()), get_i64(b.e())],
                            boolean(),
                        ))],
                    ),
                    ret(int_cmp(get_f64(a.e()), get_f64(b.e()))),
                ],
            ),
            when(
                and(is_cat(&ca, STR), is_cat(&cb, STR)),
                vec![ret(str_test(cast(
                    call("zb_str_cmp", vec![get_str(a.e()), get_str(b.e())], i32()),
                    i64(),
                )))],
            ),
            h.decl(call("zl_meta_of", vec![a.e(), text(event)], any())),
            when(
                is_nil(h.e()),
                vec![h.set(call("zl_meta_of", vec![b.e(), text(event)], any()))],
            ),
            when(
                not(is_nil(h.e())),
                vec![
                    metamethod_call_check(h.e(), text(&event[2..])),
                    ret(call(
                        "zl_truthy",
                        vec![call(
                            "zl_first",
                            vec![call("zl_call_2", vec![h.e(), a.e(), b.e()], any())],
                            any(),
                        )],
                        boolean(),
                    )),
                ],
            ),
        ];
        st.extend(compare_error(&a, &b));
        define(name, &[&a, &b], boolean(), st)
    };
    d.push(order(
        "zl_lt",
        "__lt",
        lt,
        |c| lt(c, int(0)),
        ("zl_lt_if", "zl_lt_fi"),
    ));
    d.push(order(
        "zl_le",
        "__le",
        le,
        |c| le(c, int(0)),
        ("zl_le_if", "zl_le_fi"),
    ));

    // ─── concatenation and length ───────────────────────────────
    // The text an operand of `..` contributes: strings as they are,
    // numbers as their text; anything else is the metamethod's.
    d.push(define(
        "zl_concat_text",
        &[&x],
        string(),
        vec![
            when(
                is_nil(x.e()),
                vec![
                    type_error(
                        text("attempt to concatenate a nil value"),
                        int(OPERAND_LEFT),
                    ),
                    ret(text("")),
                ],
            ),
            cat.decl(category(x.e())),
            when(is_cat(&cat, STR), vec![ret(get_str(x.e()))]),
            when(
                is_number_cat(&cat),
                vec![ret(call("zl_number_str", vec![x.e()], string()))],
            ),
            type_error(
                concat(vec![
                    text("attempt to concatenate a "),
                    type_name(x.e()),
                    text(" value"),
                ]),
                int(OPERAND_LEFT),
            ),
            ret(text("")),
        ],
    ));
    d.push(define(
        "zl_concat",
        &[&a, &b],
        any(),
        vec![
            when(
                and(
                    and(not(is_nil(a.e())), not(is_nil(b.e()))),
                    and(is_text_like(a.e()), is_text_like(b.e())),
                ),
                vec![ret(box_str(add(
                    call("zl_concat_text", vec![a.e()], string()),
                    call("zl_concat_text", vec![b.e()], string()),
                )))],
            ),
            ret(call(
                "zl_arith_meta",
                vec![int(OP_CONCAT), a.e(), b.e()],
                any(),
            )),
        ],
    ));
    // `#x`: a string's bytes, a table's border or its `__len`.
    d.push(define(
        "zl_len_any",
        &[&x],
        any(),
        vec![
            when(
                and(not(is_nil(x.e())), eq(category(x.e()), int(STR))),
                vec![ret(box_i64(call(
                    "zb_str_len",
                    vec![get_str(x.e())],
                    i64(),
                )))],
            ),
            when(
                is_table(x.e()),
                vec![
                    h.decl(call("zl_meta_of", vec![x.e(), text("__len")], any())),
                    when(
                        not(is_nil(h.e())),
                        vec![
                            metamethod_call_check(h.e(), text("len")),
                            // The operand is passed twice, as for
                            // every unary event.
                            ret(call(
                                "zl_first",
                                vec![call("zl_call_2", vec![h.e(), x.e(), x.e()], any())],
                                any(),
                            )),
                        ],
                    ),
                    ret(box_i64(call("zl_len", vec![unbox_table(x.e(), t)], i64()))),
                ],
            ),
            type_error(
                concat(vec![
                    text("attempt to get length of a "),
                    type_name(x.e()),
                    text(" value"),
                ]),
                int(OPERAND_LEFT),
            ),
            ret(nil()),
        ],
    ));
    let _ = (&i, &r);
    d
}
