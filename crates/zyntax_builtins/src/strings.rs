//! Strings: the primitives from the string plugin, and what every
//! language builds on them.

use crate::build::*;
use crate::Policy;

/// The primitives, bound to the string plugin's symbols.
const PRIMITIVES: &[(&str, &[(&str, &str)], &str, &str)] = &[
    ("zb_str_len", &[("s", "str")], "i64", "$String$length"),
    (
        "zb_str_chars_len",
        &[("s", "str")],
        "i64",
        "$String$char_count",
    ),
    (
        "zb_str_index_of",
        &[("h", "str"), ("n", "str")],
        "i64",
        "$String$index_of",
    ),
    (
        "zb_str_substring",
        &[("s", "str"), ("a", "i64"), ("b", "i64")],
        "str",
        "$String$substring",
    ),
    (
        "zb_str_char_at",
        &[("s", "str"), ("i", "i64")],
        "str",
        "$String$char_at",
    ),
    (
        "zb_str_repeat_raw",
        &[("s", "str"), ("n", "i64")],
        "str",
        "$String$repeat",
    ),
    ("zb_str_of_int", &[("n", "i64")], "str", "$String$from_int"),
    (
        "zb_str_of_float_raw",
        &[("x", "f64")],
        "str",
        "$String$from_float",
    ),
    (
        "zb_str_eq",
        &[("a", "str"), ("b", "str")],
        "bool",
        "$String$equals",
    ),
    (
        "zb_str_cmp",
        &[("a", "str"), ("b", "str")],
        "i64",
        "$String$compare",
    ),
    (
        "zb_str_contains",
        &[("h", "str"), ("n", "str")],
        "bool",
        "$String$contains",
    ),
    ("zb_str_upper", &[("s", "str")], "str", "$String$to_upper"),
    ("zb_str_lower", &[("s", "str")], "str", "$String$to_lower"),
    ("zb_str_strip", &[("s", "str")], "str", "$String$trim"),
    (
        "zb_str_lstrip",
        &[("s", "str")],
        "str",
        "$String$trim_start",
    ),
    ("zb_str_rstrip", &[("s", "str")], "str", "$String$trim_end"),
    (
        "zb_str_startswith",
        &[("s", "str"), ("p", "str")],
        "bool",
        "$String$starts_with",
    ),
    (
        "zb_str_endswith",
        &[("s", "str"), ("p", "str")],
        "bool",
        "$String$ends_with",
    ),
    (
        "zb_str_replace",
        &[("s", "str"), ("a", "str"), ("b", "str")],
        "str",
        "$String$replace_all",
    ),
    (
        "zb_str_count",
        &[("s", "str"), ("n", "str")],
        "i64",
        "$String$count",
    ),
    (
        "zb_str_parse_int",
        &[("s", "str")],
        "i64",
        "$String$parse_int",
    ),
    (
        "zb_str_parse_float",
        &[("s", "str")],
        "f64",
        "$String$parse_float",
    ),
];

fn ty(name: &str) -> zyntax_typed_ast::Type {
    match name {
        "i64" => i64(),
        "f64" => f64(),
        "bool" => boolean(),
        "str" => string(),
        _ => any(),
    }
}

fn chars_len(s: Expr) -> Expr {
    call("zb_str_chars_len", vec![s], i64())
}
fn char_at(s: Expr, i: Expr) -> Expr {
    call("zb_str_char_at", vec![s, i], string())
}
fn substring(s: Expr, a: Expr, b: Expr) -> Expr {
    call("zb_str_substring", vec![s, a, b], string())
}
fn str_eq(a: Expr, b: Expr) -> Expr {
    call("zb_str_eq", vec![a, b], boolean())
}

pub(crate) fn declarations(policy: &Policy) -> Vec<Decl> {
    let mut out = Vec::new();
    for (name, params, ret, symbol) in PRIMITIVES {
        let params: Vec<(&str, zyntax_typed_ast::Type)> =
            params.iter().map(|(n, t)| (*n, ty(t))).collect();
        out.push(extern_fn(name, &params, ty(ret), Some(symbol)));
    }
    // The compiler's own intrinsics, resolved by name.
    out.push(extern_fn("pow", &[("a", f64()), ("b", f64())], f64(), None));
    out.push(extern_fn("floor", &[("x", f64())], f64(), None));

    let s = local("s", string());
    let a = local("a", string());
    let b = local("b", string());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());

    out.push(define(
        "zb_str_truthy",
        &[&s],
        boolean(),
        vec![ret(ne(call("zb_str_len", vec![s.e()], i64()), int(0)))],
    ));
    out.push(define(
        "zb_str_lt",
        &[&a, &b],
        boolean(),
        vec![ret(lt(
            call("zb_str_cmp", vec![a.e(), b.e()], i64()),
            int(0),
        ))],
    ));
    out.push(define(
        "zb_str_repeat",
        &[&s, &n],
        string(),
        vec![
            when(le(n.e(), int(0)), vec![ret(text(""))]),
            ret(call("zb_str_repeat_raw", vec![s.e(), n.e()], string())),
        ],
    ));

    // `s[i]` with a negative index counting from the end.
    out.push(define(
        "zb_str_get",
        &[&s, &i],
        string(),
        vec![
            n.decl(chars_len(s.e())),
            j.decl(i.e()),
            when(lt(j.e(), int(0)), vec![j.set(add(j.e(), n.e()))]),
            when(
                or(lt(j.e(), int(0)), ge(j.e(), n.e())),
                vec![fatal("IndexError", text("string index out of range"))],
            ),
            ret(char_at(s.e(), j.e())),
        ],
    ));

    // A slice bound clamped the way `slice.indices` clamps it.
    let step = local("step", i64());
    out.push(define(
        "zb_slice_bound",
        &[&i, &n, &step],
        i64(),
        vec![
            j.decl(i.e()),
            when(lt(j.e(), int(0)), vec![j.set(add(j.e(), n.e()))]),
            if_(
                gt(step.e(), int(0)),
                vec![
                    when(lt(j.e(), int(0)), vec![ret(int(0))]),
                    when(gt(j.e(), n.e()), vec![ret(n.e())]),
                    ret(j.e()),
                ],
                vec![
                    when(lt(j.e(), int(0)), vec![ret(int(-1))]),
                    when(gt(j.e(), sub(n.e(), int(1))), vec![ret(sub(n.e(), int(1)))]),
                    ret(j.e()),
                ],
            ),
        ],
    ));

    // s[start:stop:step]; `mask` bits 1, 2, 4 say which bounds were given.
    let start = local("start", i64());
    let stop = local("stop", i64());
    let mask = local("mask", i64());
    let st = local("st", i64());
    let lo = local("lo", i64());
    let hi = local("hi", i64());
    let out_s = local("out", string());
    let step_body = |f: &Local| {
        vec![
            out_s.set(add(out_s.e(), char_at(s.e(), f.e()))),
            f.set(add(f.e(), st.e())),
        ]
    };
    out.push(define(
        "zb_str_slice",
        &[&s, &start, &stop, &step, &mask],
        string(),
        vec![
            n.decl(chars_len(s.e())),
            st.decl(int(1)),
            when(ne(bitand(mask.e(), int(4)), int(0)), vec![st.set(step.e())]),
            when(
                eq(st.e(), int(0)),
                vec![fatal("ValueError", text("slice step cannot be zero"))],
            ),
            lo.decl(int(0)),
            hi.decl(n.e()),
            when(
                lt(st.e(), int(0)),
                vec![lo.set(sub(n.e(), int(1))), hi.set(int(-1))],
            ),
            when(
                ne(bitand(mask.e(), int(1)), int(0)),
                vec![lo.set(call(
                    "zb_slice_bound",
                    vec![start.e(), n.e(), st.e()],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(mask.e(), int(2)), int(0)),
                vec![hi.set(call("zb_slice_bound", vec![stop.e(), n.e(), st.e()], i64()))],
            ),
            // A plain step of one is a substring.
            when(
                eq(st.e(), int(1)),
                vec![
                    when(ge(lo.e(), hi.e()), vec![ret(text(""))]),
                    ret(substring(s.e(), lo.e(), hi.e())),
                ],
            ),
            out_s.decl(text("")),
            i.decl(lo.e()),
            if_(
                gt(st.e(), int(0)),
                vec![while_(lt(i.e(), hi.e()), step_body(&i))],
                vec![while_(gt(i.e(), hi.e()), step_body(&i))],
            ),
            ret(out_s.e()),
        ],
    ));

    // repr(str): quoted with escapes, in the language's quote style.
    let quote = local("quote", string());
    let c = local("c", string());
    let (prefer, other) = if policy.single_quotes {
        ("'", "\"")
    } else {
        ("\"", "'")
    };
    let mut escapes: Vec<Stmt> = vec![out_s.set(add(out_s.e(), c.e()))];
    escapes = vec![if_(
        str_eq(c.e(), quote.e()),
        vec![out_s.set(add(add(out_s.e(), text("\\")), c.e()))],
        escapes,
    )];
    for (ch, esc) in [("\t", "\\t"), ("\n", "\\n"), ("\\", "\\\\")] {
        escapes = vec![if_(
            str_eq(c.e(), text(ch)),
            vec![out_s.set(add(out_s.e(), text(esc)))],
            escapes,
        )];
    }
    let mut repr_body = vec![quote.decl(text(prefer))];
    if policy.single_quotes {
        // A single quote inside and no double quote flips the style.
        repr_body.push(when(
            and(
                call("zb_str_contains", vec![s.e(), text("'")], boolean()),
                not(call("zb_str_contains", vec![s.e(), text("\"")], boolean())),
            ),
            vec![quote.set(text(other))],
        ));
    }
    repr_body.push(out_s.decl(quote.e()));
    repr_body.push(n.decl(chars_len(s.e())));
    let mut loop_body = vec![c.decl(char_at(s.e(), i.e()))];
    loop_body.extend(escapes);
    repr_body.extend(for_range(&i, int(0), n.e(), loop_body));
    repr_body.push(ret(add(out_s.e(), quote.e())));
    out.push(define("zb_str_repr", &[&s], string(), repr_body));

    // Casing: capitalize the first character, or the first of every word.
    let c = local("c", string());
    let lower = local("lower", string());
    let upper = local("upper", string());
    let start = local("start", boolean());
    let out_s = local("out", string());
    let to_upper = |s: Expr| call("zb_str_upper", vec![s], string());
    let to_lower = |s: Expr| call("zb_str_lower", vec![s], string());
    out.push(define(
        "zb_str_capitalize",
        &[&s],
        string(),
        vec![
            n.decl(chars_len(s.e())),
            when(eq(n.e(), int(0)), vec![ret(s.e())]),
            ret(add(
                to_upper(char_at(s.e(), int(0))),
                to_lower(substring(s.e(), int(1), n.e())),
            )),
        ],
    ));
    out.push(define("zb_str_title", &[&s], string(), {
        let mut body = vec![
            out_s.decl(text("")),
            start.decl(bool(true)),
            n.decl(chars_len(s.e())),
        ];
        body.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                c.decl(char_at(s.e(), i.e())),
                lower.decl(to_lower(c.e())),
                upper.decl(to_upper(c.e())),
                // A character without case separates words.
                if_(
                    str_eq(lower.e(), upper.e()),
                    vec![out_s.set(add(out_s.e(), c.e())), start.set(bool(true))],
                    vec![
                        if_(
                            start.e(),
                            vec![out_s.set(add(out_s.e(), upper.e()))],
                            vec![out_s.set(add(out_s.e(), lower.e()))],
                        ),
                        start.set(bool(false)),
                    ],
                ),
            ],
        ));
        body.push(ret(out_s.e()));
        body
    }));

    out.push(float_repr(policy));
    out
}

/// repr(float): the shortest digits that round-trip, positional when the
/// decimal exponent is in -4..16 and scientific otherwise, with a
/// fraction or an exponent so it reads as a float. The digits come from
/// the runtime's shortest formatting, which never uses an exponent; the
/// placement is decided here.
fn float_repr(policy: &Policy) -> Decl {
    let x = local("x", f64());
    let digits = local("digits", string());
    let sign = local("sign", string());
    let body = local("body", string());
    let dot = local("dot", i64());
    let int_part = local("int_part", string());
    let frac_part = local("frac_part", string());
    let mantissa = local("mantissa", string());
    let exponent = local("exponent", i64());
    let zeros = local("zeros", i64());
    let end = local("end", i64());
    let point = local("point", i64());
    let out = local("out", string());
    let exp_sign = local("exp_sign", string());
    let exp_abs = local("exp_abs", i64());
    let exp_digits = local("exp_digits", string());
    let zero_text = if policy.float_fraction { "0.0" } else { "0" };
    let frac = if policy.float_fraction { ".0" } else { "" };

    let mut stmts = vec![
        when(ne(x.e(), x.e()), vec![ret(text("nan"))]),
        digits.decl(call("zb_str_of_float_raw", vec![x.e()], string())),
    ];
    for inf in ["inf", "-inf"] {
        stmts.push(when(str_eq(digits.e(), text(inf)), vec![ret(text(inf))]));
    }
    stmts.extend(vec![
        sign.decl(text("")),
        body.decl(digits.e()),
        when(
            call("zb_str_startswith", vec![body.e(), text("-")], boolean()),
            vec![
                sign.set(text("-")),
                body.set(substring(body.e(), int(1), chars_len(body.e()))),
            ],
        ),
        dot.decl(call("zb_str_index_of", vec![body.e(), text(".")], i64())),
        int_part.decl(body.e()),
        frac_part.decl(text("")),
        when(
            ge(dot.e(), int(0)),
            vec![
                int_part.set(substring(body.e(), int(0), dot.e())),
                frac_part.set(substring(
                    body.e(),
                    add(dot.e(), int(1)),
                    chars_len(body.e()),
                )),
            ],
        ),
        mantissa.decl(text("")),
        exponent.decl(int(0)),
        if_(
            str_eq(int_part.e(), text("0")),
            // 0.000ddd: the exponent counts the zeros after the point.
            vec![
                zeros.decl(int(0)),
                while_(
                    and(
                        lt(zeros.e(), chars_len(frac_part.e())),
                        str_eq(char_at(frac_part.e(), zeros.e()), text("0")),
                    ),
                    vec![zeros.add_assign(int(1))],
                ),
                when(
                    eq(zeros.e(), chars_len(frac_part.e())),
                    vec![ret(add(sign.e(), text(zero_text)))],
                ),
                mantissa.set(substring(
                    frac_part.e(),
                    zeros.e(),
                    chars_len(frac_part.e()),
                )),
                exponent.set(sub(int(0), add(zeros.e(), int(1)))),
            ],
            vec![
                mantissa.set(add(int_part.e(), frac_part.e())),
                exponent.set(sub(chars_len(int_part.e()), int(1))),
            ],
        ),
        // Trailing zeros carry no information.
        end.decl(chars_len(mantissa.e())),
        while_(
            and(
                gt(end.e(), int(1)),
                str_eq(char_at(mantissa.e(), sub(end.e(), int(1))), text("0")),
            ),
            vec![end.set(sub(end.e(), int(1)))],
        ),
        mantissa.set(substring(mantissa.e(), int(0), end.e())),
        when(
            and(ge(exponent.e(), int(-4)), lt(exponent.e(), int(16))),
            vec![
                when(
                    ge(exponent.e(), int(0)),
                    vec![
                        point.decl(add(exponent.e(), int(1))),
                        when(
                            le(chars_len(mantissa.e()), point.e()),
                            vec![ret(add(
                                add(
                                    add(sign.e(), mantissa.e()),
                                    call(
                                        "zb_str_repeat",
                                        vec![text("0"), sub(point.e(), chars_len(mantissa.e()))],
                                        string(),
                                    ),
                                ),
                                text(frac),
                            ))],
                        ),
                        ret(add(
                            add(
                                add(sign.e(), substring(mantissa.e(), int(0), point.e())),
                                text("."),
                            ),
                            substring(mantissa.e(), point.e(), chars_len(mantissa.e())),
                        )),
                    ],
                ),
                ret(add(
                    add(
                        add(sign.e(), text("0.")),
                        call(
                            "zb_str_repeat",
                            vec![text("0"), sub(sub(int(0), exponent.e()), int(1))],
                            string(),
                        ),
                    ),
                    mantissa.e(),
                )),
            ],
        ),
        // Scientific: d.ddde+XX with at least two exponent digits.
        out.decl(add(sign.e(), char_at(mantissa.e(), int(0)))),
        when(
            gt(chars_len(mantissa.e()), int(1)),
            vec![out.set(add(
                add(out.e(), text(".")),
                substring(mantissa.e(), int(1), chars_len(mantissa.e())),
            ))],
        ),
        exp_sign.decl(text("+")),
        exp_abs.decl(exponent.e()),
        when(
            lt(exponent.e(), int(0)),
            vec![
                exp_sign.set(text("-")),
                exp_abs.set(sub(int(0), exponent.e())),
            ],
        ),
        exp_digits.decl(call("zb_str_of_int", vec![exp_abs.e()], string())),
        when(
            lt(exp_abs.e(), int(10)),
            vec![exp_digits.set(add(text("0"), exp_digits.e()))],
        ),
        ret(add(
            add(add(out.e(), text("e")), exp_sign.e()),
            exp_digits.e(),
        )),
    ]);
    define("zb_float_repr", &[&x], string(), stmts)
}
