//! The format mini-language: a value, a fill, an alignment, a sign
//! mode, a width, a grouping, a precision and a presentation type give
//! one string. The frontend parses a spec into these fields once; the
//! functions here apply them.
//!
//! Encodings shared with the frontend: `align` 0 unset, 1 `<`, 2 `>`,
//! 3 `^`, 4 `=`; `sign` 0 `-`, 1 `+`, 2 space; `grouping` 0 none, 1 `,`,
//! 2 `_`; `width` and `precision` -1 when unset; `ty` the presentation
//! type's character code, 0 when unset.

use crate::build::*;

const LEFT: i64 = 1;
const RIGHT: i64 = 2;
const CENTER: i64 = 3;
const AFTER_SIGN: i64 = 4;

fn chars_len(s: Expr) -> Expr {
    call("zb_str_chars_len", vec![s], i64())
}
fn substring(s: Expr, a: Expr, b: Expr) -> Expr {
    call("zb_str_substring", vec![s, a, b], string())
}
fn repeat(s: Expr, n: Expr) -> Expr {
    call("zb_str_repeat", vec![s, n], string())
}
fn upper(s: Expr) -> Expr {
    call("zb_str_upper", vec![s], string())
}
fn str_eq(a: Expr, b: Expr) -> Expr {
    call("zb_str_eq", vec![a, b], boolean())
}
fn of_int(n: Expr) -> Expr {
    call("zb_str_of_int", vec![n], string())
}
fn fixed(x: Expr, p: Expr) -> Expr {
    call("zb_str_of_float_fixed", vec![x, cast(p, i32())], string())
}
fn code(c: char) -> Expr {
    int(c as i64)
}
fn is_ty(ty: &Local, c: char) -> Expr {
    eq(ty.e(), code(c))
}
fn concat(parts: Vec<Expr>) -> Expr {
    let mut it = parts.into_iter();
    let first = it.next().expect("at least one part");
    it.fold(first, add)
}

pub(crate) fn declarations() -> Vec<Decl> {
    let mut d = vec![
        extern_fn(
            "zb_str_of_float_fixed",
            &[("x", f64()), ("p", i32())],
            string(),
            Some("$String$from_float_precision"),
        ),
        extern_fn(
            "zb_str_of_int_radix",
            &[("n", i64()), ("radix", i32())],
            string(),
            Some("$String$from_int_radix"),
        ),
    ];

    let s = local("s", string());
    let width = local("width", i64());
    let fill = local("fill", string());
    let align = local("align", i64());
    let n = local("n", i64());
    let gap = local("gap", i64());
    let left = local("left", i64());
    d.push(define(
        "zb_fmt_pad",
        &[&s, &width, &fill, &align],
        string(),
        vec![
            n.decl(chars_len(s.e())),
            when(le(width.e(), n.e()), vec![ret(s.e())]),
            gap.decl(sub(width.e(), n.e())),
            // Centring puts the odd space on the right.
            when(
                eq(align.e(), int(CENTER)),
                vec![
                    left.decl(div(gap.e(), int(2))),
                    ret(concat(vec![
                        repeat(fill.e(), left.e()),
                        s.e(),
                        repeat(fill.e(), sub(gap.e(), left.e())),
                    ])),
                ],
            ),
            when(
                eq(align.e(), int(LEFT)),
                vec![ret(add(s.e(), repeat(fill.e(), gap.e())))],
            ),
            ret(add(repeat(fill.e(), gap.e()), s.e())),
        ],
    ));

    // Digit grouping from the right, in threes.
    let digits = local("digits", string());
    let sep = local("sep", string());
    let out = local("out", string());
    let i = local("i", i64());
    let head = local("head", i64());
    d.push(define(
        "zb_fmt_group",
        &[&digits, &sep],
        string(),
        vec![
            n.decl(chars_len(digits.e())),
            when(le(n.e(), int(3)), vec![ret(digits.e())]),
            head.decl(rem(n.e(), int(3))),
            when(eq(head.e(), int(0)), vec![head.set(int(3))]),
            out.decl(substring(digits.e(), int(0), head.e())),
            i.decl(head.e()),
            while_(
                lt(i.e(), n.e()),
                vec![
                    out.set(concat(vec![
                        out.e(),
                        sep.e(),
                        substring(digits.e(), i.e(), add(i.e(), int(3))),
                    ])),
                    i.add_assign(int(3)),
                ],
            ),
            ret(out.e()),
        ],
    ));
    let grouping = local("grouping", i64());
    d.push(define(
        "zb_fmt_grouped",
        &[&digits, &grouping],
        string(),
        vec![
            when(
                eq(grouping.e(), int(1)),
                vec![ret(call(
                    "zb_fmt_group",
                    vec![digits.e(), text(",")],
                    string(),
                ))],
            ),
            when(
                eq(grouping.e(), int(2)),
                vec![ret(call(
                    "zb_fmt_group",
                    vec![digits.e(), text("_")],
                    string(),
                ))],
            ),
            ret(digits.e()),
        ],
    ));

    // A magnitude with its sign, padded: zero padding and `=` alignment
    // go between the sign and the digits, anything else around both.
    let body = local("body", string());
    let negative = local("negative", boolean());
    let sign = local("sign", i64());
    let zero = local("zero", boolean());
    let prefix = local("prefix", string());
    let pad_fill = local("pad_fill", string());
    let a = local("a", i64());
    d.push(define(
        "zb_fmt_signed",
        &[&body, &negative, &sign, &zero, &fill, &align, &width],
        string(),
        vec![
            prefix.decl(text("")),
            if_(
                negative.e(),
                vec![prefix.set(text("-"))],
                vec![
                    when(eq(sign.e(), int(1)), vec![prefix.set(text("+"))]),
                    when(eq(sign.e(), int(2)), vec![prefix.set(text(" "))]),
                ],
            ),
            when(
                or(
                    eq(align.e(), int(AFTER_SIGN)),
                    and(zero.e(), eq(align.e(), int(0))),
                ),
                vec![
                    pad_fill.decl(fill.e()),
                    when(
                        and(zero.e(), eq(align.e(), int(0))),
                        vec![pad_fill.set(text("0"))],
                    ),
                    n.decl(add(chars_len(prefix.e()), chars_len(body.e()))),
                    when(
                        gt(width.e(), n.e()),
                        vec![ret(concat(vec![
                            prefix.e(),
                            repeat(pad_fill.e(), sub(width.e(), n.e())),
                            body.e(),
                        ]))],
                    ),
                    ret(add(prefix.e(), body.e())),
                ],
            ),
            a.decl(align.e()),
            when(eq(a.e(), int(0)), vec![a.set(int(RIGHT))]),
            ret(call(
                "zb_fmt_pad",
                vec![add(prefix.e(), body.e()), width.e(), fill.e(), a.e()],
                string(),
            )),
        ],
    ));

    d.extend(int_format());
    d.extend(float_format());

    // Strings: truncated to the precision, left-aligned by default.
    let precision = local("precision", i64());
    let text_in = local("s", string());
    d.push(define(
        "zb_fmt_str",
        &[&text_in, &fill, &align, &width, &precision],
        string(),
        vec![
            out.decl(text_in.e()),
            when(
                and(
                    ge(precision.e(), int(0)),
                    lt(precision.e(), chars_len(out.e())),
                ),
                vec![out.set(substring(out.e(), int(0), precision.e()))],
            ),
            a.decl(align.e()),
            when(eq(a.e(), int(0)), vec![a.set(int(LEFT))]),
            ret(call(
                "zb_fmt_pad",
                vec![out.e(), width.e(), fill.e(), a.e()],
                string(),
            )),
        ],
    ));

    // A dynamic value formats as what it holds.
    let x = local("x", any());
    let alt = local("alt", boolean());
    let ty = local("ty", i64());
    let cat = local("cat", i64());
    let float_ty = || {
        let mut e = is_ty(&ty, 'f');
        for c in ['F', 'e', 'E', 'g', 'G', '%'] {
            e = or(e, is_ty(&ty, c));
        }
        e
    };
    d.push(define(
        "zb_any_fmt",
        &[
            &x, &fill, &align, &sign, &alt, &zero, &width, &grouping, &precision, &ty,
        ],
        string(),
        vec![
            cat.decl(call("zb_any_category", vec![x.e()], i64())),
            when(
                and(ge(cat.e(), int(1)), le(cat.e(), int(3))),
                vec![
                    when(
                        float_ty(),
                        vec![ret(call(
                            "zb_fmt_float",
                            vec![
                                call("zb_number_f64", vec![x.e(), cat.e()], f64()),
                                fill.e(),
                                align.e(),
                                sign.e(),
                                zero.e(),
                                width.e(),
                                grouping.e(),
                                precision.e(),
                                ty.e(),
                            ],
                            string(),
                        ))],
                    ),
                    // A bool with no presentation type prints its name.
                    when(
                        and(eq(cat.e(), int(1)), eq(ty.e(), int(0))),
                        vec![ret(call(
                            "zb_fmt_str",
                            vec![
                                call("zb_any_str", vec![x.e()], string()),
                                fill.e(),
                                align.e(),
                                width.e(),
                                precision.e(),
                            ],
                            string(),
                        ))],
                    ),
                    ret(call(
                        "zb_fmt_int",
                        vec![
                            call("zb_number_i64", vec![x.e(), cat.e()], i64()),
                            fill.e(),
                            align.e(),
                            sign.e(),
                            alt.e(),
                            zero.e(),
                            width.e(),
                            grouping.e(),
                            ty.e(),
                        ],
                        string(),
                    )),
                ],
            ),
            when(
                eq(cat.e(), int(4)),
                vec![ret(call(
                    "zb_fmt_float",
                    vec![
                        call("zb_box_get_f64", vec![x.e()], f64()),
                        fill.e(),
                        align.e(),
                        sign.e(),
                        zero.e(),
                        width.e(),
                        grouping.e(),
                        precision.e(),
                        ty.e(),
                    ],
                    string(),
                ))],
            ),
            ret(call(
                "zb_fmt_str",
                vec![
                    call("zb_any_str", vec![x.e()], string()),
                    fill.e(),
                    align.e(),
                    width.e(),
                    precision.e(),
                ],
                string(),
            )),
        ],
    ));
    d
}

fn int_format() -> Vec<Decl> {
    let v = local("v", i64());
    let fill = local("fill", string());
    let align = local("align", i64());
    let sign = local("sign", i64());
    let alt = local("alt", boolean());
    let zero = local("zero", boolean());
    let width = local("width", i64());
    let grouping = local("grouping", i64());
    let ty = local("ty", i64());
    let negative = local("negative", boolean());
    let mag = local("mag", i64());
    let body = local("body", string());
    let radix = |r: i32| call("zb_str_of_int_radix", vec![mag.e(), int32(r)], string());
    let with_prefix = |p: &str, digits: Expr| {
        vec![
            body.set(digits),
            when(alt.e(), vec![body.set(add(text(p), body.e()))]),
        ]
    };
    vec![define(
        "zb_fmt_int",
        &[
            &v, &fill, &align, &sign, &alt, &zero, &width, &grouping, &ty,
        ],
        string(),
        vec![
            negative.decl(lt(v.e(), int(0))),
            mag.decl(v.e()),
            when(negative.e(), vec![mag.set(sub(int(0), v.e()))]),
            body.decl(text("")),
            if_(
                is_ty(&ty, 'x'),
                with_prefix("0x", radix(16)),
                vec![if_(
                    is_ty(&ty, 'X'),
                    with_prefix("0X", upper(radix(16))),
                    vec![if_(
                        is_ty(&ty, 'o'),
                        with_prefix("0o", radix(8)),
                        vec![if_(
                            is_ty(&ty, 'b'),
                            with_prefix("0b", radix(2)),
                            vec![body.set(call(
                                "zb_fmt_grouped",
                                vec![of_int(mag.e()), grouping.e()],
                                string(),
                            ))],
                        )],
                    )],
                )],
            ),
            ret(call(
                "zb_fmt_signed",
                vec![
                    body.e(),
                    negative.e(),
                    sign.e(),
                    zero.e(),
                    fill.e(),
                    align.e(),
                    width.e(),
                ],
                string(),
            )),
        ],
    )]
}

fn float_format() -> Vec<Decl> {
    let v = local("v", f64());
    let fill = local("fill", string());
    let align = local("align", i64());
    let sign = local("sign", i64());
    let zero = local("zero", boolean());
    let width = local("width", i64());
    let grouping = local("grouping", i64());
    let precision = local("precision", i64());
    let ty = local("ty", i64());
    let negative = local("negative", boolean());
    let mag = local("mag", f64());
    let p = local("p", i64());
    let body = local("body", string());
    let dot = local("dot", i64());
    let mut d = Vec::new();

    // d.ddd with `p` decimals, times ten to `exp`: the mantissa is
    // brought into [1, 10) and rounded, carrying into the exponent when
    // rounding reaches 10.
    let x = local("x", f64());
    let m = local("m", f64());
    let exp = local("exp", i64());
    let mant = local("mant", string());
    let exp_text = local("exp_text", string());
    let e_char = local("e_char", string());
    d.push(define(
        "zb_fmt_exponent",
        &[&x, &p, &e_char],
        string(),
        vec![
            m.decl(x.e()),
            exp.decl(int(0)),
            when(
                gt(m.e(), float(0.0)),
                vec![
                    while_(
                        ge(m.e(), float(10.0)),
                        vec![m.set(div(m.e(), float(10.0))), exp.add_assign(int(1))],
                    ),
                    while_(
                        lt(m.e(), float(1.0)),
                        vec![
                            m.set(mul(m.e(), float(10.0))),
                            exp.set(sub(exp.e(), int(1))),
                        ],
                    ),
                ],
            ),
            mant.decl(fixed(m.e(), p.e())),
            when(
                call("zb_str_startswith", vec![mant.e(), text("10")], boolean()),
                vec![
                    mant.set(fixed(div(m.e(), float(10.0)), p.e())),
                    exp.add_assign(int(1)),
                ],
            ),
            exp_text.decl(text("+")),
            when(
                lt(exp.e(), int(0)),
                vec![exp_text.set(text("-")), exp.set(sub(int(0), exp.e()))],
            ),
            when(
                lt(exp.e(), int(10)),
                vec![exp_text.set(add(exp_text.e(), text("0")))],
            ),
            ret(concat(vec![
                mant.e(),
                e_char.e(),
                exp_text.e(),
                of_int(exp.e()),
            ])),
        ],
    ));

    // Trailing zeros of a fraction, and a bare point, carry nothing.
    let s = local("s", string());
    let end = local("end", i64());
    d.push(define(
        "zb_fmt_strip_zeros",
        &[&s],
        string(),
        vec![
            dot.decl(call("zb_str_index_of", vec![s.e(), text(".")], i64())),
            when(lt(dot.e(), int(0)), vec![ret(s.e())]),
            end.decl(call("zb_str_chars_len", vec![s.e()], i64())),
            while_(
                and(
                    gt(end.e(), add(dot.e(), int(1))),
                    str_eq(
                        call(
                            "zb_str_char_at",
                            vec![s.e(), sub(end.e(), int(1))],
                            string(),
                        ),
                        text("0"),
                    ),
                ),
                vec![end.set(sub(end.e(), int(1)))],
            ),
            when(eq(end.e(), add(dot.e(), int(1))), vec![end.set(dot.e())]),
            ret(substring(s.e(), int(0), end.e())),
        ],
    ));

    // General: `p` significant digits, positional when the exponent is
    // in [-4, p), scientific otherwise. `keep_point` is the untyped
    // form, which still reads as a float.
    let keep_point = local("keep_point", boolean());
    let exp_of = local("exp_of", i64());
    let probe = local("probe", string());
    let e_pos = local("e_pos", i64());
    d.push(define(
        "zb_fmt_general",
        &[&x, &p, &keep_point, &e_char],
        string(),
        vec![
            when(le(p.e(), int(0)), vec![p.set(int(1))]),
            // The exponent after rounding to p digits comes from the
            // scientific form itself.
            probe.decl(call(
                "zb_fmt_exponent",
                vec![x.e(), sub(p.e(), int(1)), text("e")],
                string(),
            )),
            e_pos.decl(call("zb_str_index_of", vec![probe.e(), text("e")], i64())),
            exp_of.decl(call(
                "zb_str_parse_int",
                vec![substring(
                    probe.e(),
                    add(e_pos.e(), int(1)),
                    chars_len(probe.e()),
                )],
                i64(),
            )),
            body.decl(text("")),
            if_(
                and(ge(exp_of.e(), int(-4)), lt(exp_of.e(), p.e())),
                vec![body.set(call(
                    "zb_fmt_strip_zeros",
                    vec![fixed(x.e(), sub(sub(p.e(), int(1)), exp_of.e()))],
                    string(),
                ))],
                vec![body.set(concat(vec![
                    call(
                        "zb_fmt_strip_zeros",
                        vec![substring(probe.e(), int(0), e_pos.e())],
                        string(),
                    ),
                    e_char.e(),
                    substring(probe.e(), add(e_pos.e(), int(1)), chars_len(probe.e())),
                ]))],
            ),
            when(
                and(
                    keep_point.e(),
                    and(
                        lt(
                            call("zb_str_index_of", vec![body.e(), text(".")], i64()),
                            int(0),
                        ),
                        lt(
                            call("zb_str_index_of", vec![body.e(), text("e")], i64()),
                            int(0),
                        ),
                    ),
                ),
                vec![body.set(add(body.e(), text(".0")))],
            ),
            ret(body.e()),
        ],
    ));

    let int_part = local("int_part", string());
    d.push(define(
        "zb_fmt_float",
        &[
            &v, &fill, &align, &sign, &zero, &width, &grouping, &precision, &ty,
        ],
        string(),
        vec![
            negative.decl(call(
                "zb_str_startswith",
                vec![
                    call("zb_str_of_float_raw", vec![v.e()], string()),
                    text("-"),
                ],
                boolean(),
            )),
            mag.decl(v.e()),
            when(negative.e(), vec![mag.set(sub(float(0.0), v.e()))]),
            p.decl(precision.e()),
            when(lt(p.e(), int(0)), vec![p.set(int(6))]),
            body.decl(text("")),
            if_(
                ne(mag.e(), mag.e()),
                vec![body.set(text("nan"))],
                vec![if_(
                    str_eq(
                        call("zb_str_of_float_raw", vec![mag.e()], string()),
                        text("inf"),
                    ),
                    vec![body.set(text("inf"))],
                    vec![if_(
                        or(is_ty(&ty, 'f'), is_ty(&ty, 'F')),
                        vec![body.set(fixed(mag.e(), p.e()))],
                        vec![if_(
                            is_ty(&ty, '%'),
                            vec![
                                body.set(add(fixed(mul(mag.e(), float(100.0)), p.e()), text("%"))),
                            ],
                            vec![if_(
                                is_ty(&ty, 'e'),
                                vec![body.set(call(
                                    "zb_fmt_exponent",
                                    vec![mag.e(), p.e(), text("e")],
                                    string(),
                                ))],
                                vec![if_(
                                    is_ty(&ty, 'E'),
                                    vec![body.set(call(
                                        "zb_fmt_exponent",
                                        vec![mag.e(), p.e(), text("E")],
                                        string(),
                                    ))],
                                    vec![if_(
                                        is_ty(&ty, 'g'),
                                        vec![body.set(call(
                                            "zb_fmt_general",
                                            vec![mag.e(), p.e(), bool(false), text("e")],
                                            string(),
                                        ))],
                                        vec![if_(
                                            is_ty(&ty, 'G'),
                                            vec![body.set(call(
                                                "zb_fmt_general",
                                                vec![mag.e(), p.e(), bool(false), text("E")],
                                                string(),
                                            ))],
                                            vec![if_(
                                                // No type: repr, or `g`
                                                // that keeps its point
                                                // when a precision is given.
                                                lt(precision.e(), int(0)),
                                                vec![body.set(call(
                                                    "zb_float_repr",
                                                    vec![mag.e()],
                                                    string(),
                                                ))],
                                                vec![body.set(call(
                                                    "zb_fmt_general",
                                                    vec![mag.e(), p.e(), bool(true), text("e")],
                                                    string(),
                                                ))],
                                            )],
                                        )],
                                    )],
                                )],
                            )],
                        )],
                    )],
                )],
            ),
            // Grouping applies to the integer digits only.
            when(
                ne(grouping.e(), int(0)),
                vec![
                    dot.decl(call("zb_str_index_of", vec![body.e(), text(".")], i64())),
                    when(lt(dot.e(), int(0)), vec![dot.set(chars_len(body.e()))]),
                    int_part.decl(substring(body.e(), int(0), dot.e())),
                    body.set(add(
                        call("zb_fmt_grouped", vec![int_part.e(), grouping.e()], string()),
                        substring(body.e(), dot.e(), chars_len(body.e())),
                    )),
                ],
            ),
            ret(call(
                "zb_fmt_signed",
                vec![
                    body.e(),
                    negative.e(),
                    sign.e(),
                    zero.e(),
                    fill.e(),
                    align.e(),
                    width.e(),
                ],
                string(),
            )),
        ],
    ));
    d
}
