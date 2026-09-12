//! Dynamic values. An `Any` is a box with a type tag; the low byte of
//! the tag is the category (0 none, 1 bool, 2 signed int, 3 unsigned
//! int, 4 float, 5 string, 255 custom) and, for a boxed list or tuple,
//! the kind sits in the byte above. Every operation switches on the
//! category and then does what the typed code does.

use crate::build::*;
use crate::{list_of, Kind, Policy, DICT_TAG, FUNC_TAG, INSTANCE_KIND_BASE, SET_TAG, TUPLE_TAG};
use zyntax_typed_ast::TypeId;

const NONE: i64 = 0;
const BOOL: i64 = 1;
const INT: i64 = 2;
const UINT: i64 = 3;
const FLOAT: i64 = 4;
const STR: i64 = 5;
const CUSTOM: i64 = 255;

fn tag(x: Expr) -> Expr {
    cast(call("zb_box_tag", vec![x], i32()), i64())
}
fn category(x: Expr) -> Expr {
    call("zb_any_category", vec![x], i64())
}
fn kind(x: Expr) -> Expr {
    call("zb_any_kind", vec![x], i64())
}
fn type_name(x: Expr) -> Expr {
    call("zb_any_type", vec![x], string())
}
fn box_i64(v: Expr) -> Expr {
    call("zb_box_i64", vec![v], any())
}
fn box_f64(v: Expr) -> Expr {
    call("zb_box_f64", vec![v], any())
}
fn box_str(v: Expr) -> Expr {
    call("zb_box_str", vec![v], any())
}
fn get_i64(x: Expr) -> Expr {
    call("zb_box_get_i64", vec![x], i64())
}
fn get_f64(x: Expr) -> Expr {
    call("zb_box_get_f64", vec![x], f64())
}
fn get_bool(x: Expr) -> Expr {
    ne(call("zb_box_get_bool", vec![x], i32()), int32(0))
}
fn get_str(x: Expr) -> Expr {
    call("zb_box_get_str", vec![x], string())
}
fn is_number(cat: Expr) -> Expr {
    call("zb_is_number", vec![cat], boolean())
}
fn number_f64(x: Expr, cat: Expr) -> Expr {
    call("zb_number_f64", vec![x, cat], f64())
}
fn number_i64(x: Expr, cat: Expr) -> Expr {
    call("zb_number_i64", vec![x, cat], i64())
}
fn str_eq(a: Expr, b: Expr) -> Expr {
    call("zb_str_eq", vec![a, b], boolean())
}
fn is(cat: &Local, c: i64) -> Expr {
    eq(cat.e(), int(c))
}
fn is_int(cat: &Local) -> Expr {
    or(is(cat, INT), is(cat, UINT))
}
fn is_integral(cat: &Local) -> Expr {
    or(is(cat, BOOL), is_int(cat))
}
/// Whether a custom-category box holds an instance of a frontend class.
fn is_instance(x: Expr) -> Expr {
    ge(kind(x), int(INSTANCE_KIND_BASE))
}

/// What the library says about instances when the frontend defines no
/// hooks: nothing is equal to anything, and they print as objects.
pub(crate) fn default_instance_hooks(policy: &Policy) -> Vec<Decl> {
    let x = local("x", any());
    let a = local("a", any());
    let b = local("b", any());
    vec![
        define(
            "zb_hook_instance_str",
            &[&x],
            string(),
            vec![ret(text(&format!("<{}>", policy.type_names.object)))],
        ),
        define(
            "zb_hook_instance_type",
            &[&x],
            string(),
            vec![ret(text(policy.type_names.object))],
        ),
        define(
            "zb_hook_instance_eq",
            &[&a, &b],
            boolean(),
            vec![ret(bool(false))],
        ),
    ]
}

fn quoted(x: Expr) -> Expr {
    add(add(text("'"), x), text("'"))
}
fn type_error(message: Expr) -> Stmt {
    fatal("TypeError", message)
}

pub(crate) fn declarations(policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let x = local("x", any());
    let a = local("a", any());
    let b = local("b", any());
    let cat = local("cat", i64());
    let ca = local("ca", i64());
    let cb = local("cb", i64());
    let mut d = Vec::new();

    // An instance travels as its address under the class's tag.
    d.push(extern_fn(
        "zb_box_instance_raw",
        &[("p", i64()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    d.push(extern_fn(
        "zb_unbox_instance_raw",
        &[("x", any())],
        i64(),
        Some("zyntax_box_get_opaque"),
    ));
    // The box accessors.
    d.push(extern_fn(
        "zb_box_tag",
        &[("x", any())],
        i32(),
        Some("zyntax_box_get_tag"),
    ));
    d.push(extern_fn(
        "zb_box_get_i64",
        &[("x", any())],
        i64(),
        Some("zyntax_box_get_i64"),
    ));
    d.push(extern_fn(
        "zb_box_get_f64",
        &[("x", any())],
        f64(),
        Some("zyntax_box_get_f64"),
    ));
    d.push(extern_fn(
        "zb_box_get_bool",
        &[("x", any())],
        i32(),
        Some("zyntax_box_get_bool"),
    ));
    d.push(extern_fn(
        "zb_box_get_str",
        &[("x", any())],
        string(),
        Some("zyntax_box_get_opaque"),
    ));

    // A typed value becomes a dynamic one by being returned as one.
    for (name, ty) in [
        ("zb_box_i64", i64()),
        ("zb_box_f64", f64()),
        ("zb_box_bool", boolean()),
        ("zb_box_str", string()),
    ] {
        let v = local("v", ty);
        d.push(define(name, &[&v], any(), vec![ret(v.e())]));
    }

    d.push(define(
        "zb_any_category",
        &[&x],
        i64(),
        vec![ret(bitand(tag(x.e()), int(255)))],
    ));
    d.push(define(
        "zb_any_kind",
        &[&x],
        i64(),
        vec![ret(shr(tag(x.e()), int(8)))],
    ));
    d.push(define(
        "zb_is_number",
        &[&cat],
        boolean(),
        vec![ret(and(ge(cat.e(), int(BOOL)), le(cat.e(), int(FLOAT))))],
    ));
    d.push(define(
        "zb_number_f64",
        &[&x, &cat],
        f64(),
        vec![
            when(is(&cat, FLOAT), vec![ret(get_f64(x.e()))]),
            when(
                is(&cat, BOOL),
                vec![
                    when(get_bool(x.e()), vec![ret(float(1.0))]),
                    ret(float(0.0)),
                ],
            ),
            ret(cast(get_i64(x.e()), f64())),
        ],
    ));
    d.push(define(
        "zb_number_i64",
        &[&x, &cat],
        i64(),
        vec![
            when(
                is(&cat, BOOL),
                vec![when(get_bool(x.e()), vec![ret(int(1))]), ret(int(0))],
            ),
            ret(get_i64(x.e())),
        ],
    ));

    // Spellings.
    let flag = local("b", boolean());
    d.push(define(
        "zb_bool_repr",
        &[&flag],
        string(),
        vec![
            when(flag.e(), vec![ret(text(policy.true_text))]),
            ret(text(policy.false_text)),
        ],
    ));
    d.push(define(
        "zb_none_repr",
        &[],
        string(),
        vec![ret(text(policy.none_text))],
    ));
    let names = &policy.type_names;
    d.push(define(
        "zb_any_type",
        &[&x],
        string(),
        vec![
            cat.decl(category(x.e())),
            when(is(&cat, NONE), vec![ret(text(names.none))]),
            when(is(&cat, BOOL), vec![ret(text(names.bool))]),
            when(is_int(&cat), vec![ret(text(names.int))]),
            when(is(&cat, FLOAT), vec![ret(text(names.float))]),
            when(is(&cat, STR), vec![ret(text(names.str))]),
            when(
                is(&cat, CUSTOM),
                vec![
                    when(
                        is_instance(x.e()),
                        vec![ret(call("zb_hook_instance_type", vec![x.e()], string()))],
                    ),
                    when(
                        eq(kind(x.e()), int(TUPLE_TAG >> 8)),
                        vec![ret(text(names.tuple))],
                    ),
                    when(
                        eq(kind(x.e()), int(FUNC_TAG >> 8)),
                        vec![ret(text(names.function))],
                    ),
                    when(
                        eq(kind(x.e()), int(DICT_TAG >> 8)),
                        vec![ret(text(names.dict))],
                    ),
                    when(
                        eq(kind(x.e()), int(SET_TAG >> 8)),
                        vec![ret(text(names.set))],
                    ),
                    ret(text(names.list)),
                ],
            ),
            ret(text(names.object)),
        ],
    ));

    d.push(define(
        "zb_any_truthy",
        &[&x],
        boolean(),
        vec![
            cat.decl(category(x.e())),
            when(is(&cat, NONE), vec![ret(bool(false))]),
            when(is(&cat, BOOL), vec![ret(get_bool(x.e()))]),
            when(is_int(&cat), vec![ret(ne(get_i64(x.e()), int(0)))]),
            when(is(&cat, FLOAT), vec![ret(ne(get_f64(x.e()), float(0.0)))]),
            when(
                is(&cat, STR),
                vec![ret(call("zb_str_truthy", vec![get_str(x.e())], boolean()))],
            ),
            when(
                and(
                    is(&cat, CUSTOM),
                    and(ne(kind(x.e()), int(FUNC_TAG >> 8)), not(is_instance(x.e()))),
                ),
                vec![ret(ne(call("zb_seq_len_any", vec![x.e()], i64()), int(0)))],
            ),
            ret(bool(true)),
        ],
    ));
    d.push(define(
        "zb_any_str",
        &[&x],
        string(),
        vec![
            cat.decl(category(x.e())),
            when(is(&cat, NONE), vec![ret(text(policy.none_text))]),
            when(
                is(&cat, BOOL),
                vec![ret(call("zb_bool_repr", vec![get_bool(x.e())], string()))],
            ),
            when(
                is_int(&cat),
                vec![ret(call("zb_str_of_int", vec![get_i64(x.e())], string()))],
            ),
            when(
                is(&cat, FLOAT),
                vec![ret(call("zb_float_repr", vec![get_f64(x.e())], string()))],
            ),
            when(is(&cat, STR), vec![ret(get_str(x.e()))]),
            when(
                is(&cat, CUSTOM),
                vec![ret(call("zb_seq_repr_any", vec![x.e()], string()))],
            ),
            ret(call("zb_format_dynamic", vec![x.e()], string())),
        ],
    ));
    d.push(define(
        "zb_any_repr",
        &[&x],
        string(),
        vec![
            when(
                eq(category(x.e()), int(STR)),
                vec![ret(call("zb_str_repr", vec![get_str(x.e())], string()))],
            ),
            ret(call("zb_any_str", vec![x.e()], string())),
        ],
    ));

    // Equality and ordering across categories: numbers compare by value,
    // strings by content, sequences elementwise, and nothing else is
    // equal to anything.
    let iter = |x: Expr| call("zb_any_iter", vec![x], anys.clone());
    let is_tuple = |x: Expr| eq(kind(x), int(TUPLE_TAG >> 8));
    let is_dict = |x: Expr| eq(kind(x), int(DICT_TAG >> 8));
    let is_set = |x: Expr| eq(kind(x), int(SET_TAG >> 8));
    let raw_any = |x: Expr| call("zb_unbox_list_raw_any", vec![x], anys.clone());
    d.push(define(
        "zb_any_eq",
        &[&a, &b],
        boolean(),
        vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                or(is(&ca, NONE), is(&cb, NONE)),
                vec![ret(eq(ca.e(), cb.e()))],
            ),
            when(
                or(is(&ca, STR), is(&cb, STR)),
                vec![
                    when(
                        and(is(&ca, STR), is(&cb, STR)),
                        vec![ret(str_eq(get_str(a.e()), get_str(b.e())))],
                    ),
                    ret(bool(false)),
                ],
            ),
            when(
                and(is_number(ca.e()), is_number(cb.e())),
                vec![
                    when(
                        or(is(&ca, FLOAT), is(&cb, FLOAT)),
                        vec![ret(eq(
                            number_f64(a.e(), ca.e()),
                            number_f64(b.e(), cb.e()),
                        ))],
                    ),
                    ret(eq(number_i64(a.e(), ca.e()), number_i64(b.e(), cb.e()))),
                ],
            ),
            when(
                or(
                    and(is(&ca, CUSTOM), is_instance(a.e())),
                    and(is(&cb, CUSTOM), is_instance(b.e())),
                ),
                vec![ret(call(
                    "zb_hook_instance_eq",
                    vec![a.e(), b.e()],
                    boolean(),
                ))],
            ),
            when(
                and(is(&ca, CUSTOM), is(&cb, CUSTOM)),
                vec![
                    when(
                        ne(kind(a.e()), kind(b.e())),
                        vec![
                            // Lists of different element kinds still compare
                            // as lists; anything else differs by kind.
                            when(
                                or(
                                    or(is_tuple(a.e()), is_tuple(b.e())),
                                    or(
                                        or(is_dict(a.e()), is_dict(b.e())),
                                        or(is_set(a.e()), is_set(b.e())),
                                    ),
                                ),
                                vec![ret(bool(false))],
                            ),
                        ],
                    ),
                    when(
                        is_dict(a.e()),
                        vec![ret(call(
                            "zb_dict_eq",
                            vec![raw_any(a.e()), raw_any(b.e())],
                            boolean(),
                        ))],
                    ),
                    when(
                        is_set(a.e()),
                        vec![ret(call(
                            "zb_set_eq",
                            vec![raw_any(a.e()), raw_any(b.e())],
                            boolean(),
                        ))],
                    ),
                    ret(call(
                        "zb_list_eq_any",
                        vec![iter(a.e()), iter(b.e())],
                        boolean(),
                    )),
                ],
            ),
            ret(bool(false)),
        ],
    ));
    d.push(define(
        "zb_any_lt",
        &[&a, &b],
        boolean(),
        vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                and(is(&ca, STR), is(&cb, STR)),
                vec![ret(call(
                    "zb_str_lt",
                    vec![get_str(a.e()), get_str(b.e())],
                    boolean(),
                ))],
            ),
            when(
                and(is_number(ca.e()), is_number(cb.e())),
                vec![
                    when(
                        or(is(&ca, FLOAT), is(&cb, FLOAT)),
                        vec![ret(lt(
                            number_f64(a.e(), ca.e()),
                            number_f64(b.e(), cb.e()),
                        ))],
                    ),
                    ret(lt(number_i64(a.e(), ca.e()), number_i64(b.e(), cb.e()))),
                ],
            ),
            when(
                and(is(&ca, CUSTOM), is(&cb, CUSTOM)),
                vec![ret(call(
                    "zb_list_lt_any",
                    vec![iter(a.e()), iter(b.e())],
                    boolean(),
                ))],
            ),
            type_error(add(
                add(
                    add(
                        text("'<' not supported between instances of "),
                        quoted(type_name(a.e())),
                    ),
                    text(" and "),
                ),
                quoted(type_name(b.e())),
            )),
            ret(bool(false)),
        ],
    ));
    // Two boxed heap objects are the same when they hold the same
    // address; a box's own address means nothing, boxes are made freely.
    d.push(define(
        "zb_any_same",
        &[&a, &b],
        boolean(),
        vec![
            when(
                or(
                    ne(category(a.e()), int(CUSTOM)),
                    ne(category(b.e()), int(CUSTOM)),
                ),
                vec![ret(call("zb_any_is", vec![a.e(), b.e()], boolean()))],
            ),
            ret(eq(
                cast(raw_any(a.e()), usize()),
                cast(raw_any(b.e()), usize()),
            )),
        ],
    ));
    // Identity: the singletons compare by category, heap objects by
    // address, everything else by value.
    d.push(define(
        "zb_any_is",
        &[&a, &b],
        boolean(),
        vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                or(is(&ca, NONE), is(&cb, NONE)),
                vec![ret(eq(ca.e(), cb.e()))],
            ),
            when(
                and(is(&ca, CUSTOM), is(&cb, CUSTOM)),
                vec![ret(eq(
                    cast(raw_any(a.e()), usize()),
                    cast(raw_any(b.e()), usize()),
                ))],
            ),
            when(
                and(is(&ca, BOOL), is(&cb, BOOL)),
                vec![ret(eq(get_bool(a.e()), get_bool(b.e())))],
            ),
            ret(call("zb_any_eq", vec![a.e(), b.e()], boolean())),
        ],
    ));
    let container = local("container", any());
    let item = local("item", any());
    d.push(define(
        "zb_any_contains",
        &[&container, &item],
        boolean(),
        vec![
            cat.decl(category(container.e())),
            when(
                and(is(&cat, STR), eq(category(item.e()), int(STR))),
                vec![ret(call(
                    "zb_str_contains",
                    vec![get_str(container.e()), get_str(item.e())],
                    boolean(),
                ))],
            ),
            when(
                is(&cat, CUSTOM),
                vec![ret(call(
                    "zb_list_contains_any",
                    vec![iter(container.e()), item.e()],
                    boolean(),
                ))],
            ),
            type_error(add(
                add(text("argument of type "), quoted(type_name(container.e()))),
                text(" is not iterable"),
            )),
            ret(bool(false)),
        ],
    ));
    d.push(define(
        "zb_any_len",
        &[&x],
        i64(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(call("zb_str_chars_len", vec![get_str(x.e())], i64()))],
            ),
            when(
                and(is(&cat, CUSTOM), not(is_instance(x.e()))),
                vec![ret(call("zb_seq_len_any", vec![x.e()], i64()))],
            ),
            type_error(add(
                add(text("object of type "), quoted(type_name(x.e()))),
                text(" has no len()"),
            )),
            ret(int(0)),
        ],
    ));

    // Conversions.
    d.push(define(
        "zb_any_int",
        &[&x],
        i64(),
        vec![
            cat.decl(category(x.e())),
            when(is(&cat, FLOAT), vec![ret(cast(get_f64(x.e()), i64()))]),
            when(
                is(&cat, STR),
                vec![ret(call("zb_str_parse_int", vec![get_str(x.e())], i64()))],
            ),
            ret(number_i64(x.e(), cat.e())),
        ],
    ));
    d.push(define(
        "zb_any_float",
        &[&x],
        f64(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(call("zb_str_parse_float", vec![get_str(x.e())], f64()))],
            ),
            ret(number_f64(x.e(), cat.e())),
        ],
    ));
    d.push(define(
        "zb_any_neg",
        &[&x],
        any(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, FLOAT),
                vec![ret(box_f64(sub(float(0.0), get_f64(x.e()))))],
            ),
            ret(box_i64(sub(int(0), number_i64(x.e(), cat.e())))),
        ],
    ));
    let kept = owned("x", any());
    d.push(define("zb_any_pos", &[&kept], any(), vec![ret(kept.e())]));
    d.push(define(
        "zb_any_invert",
        &[&x],
        any(),
        vec![ret(box_i64(sub(
            int(-1),
            number_i64(x.e(), category(x.e())),
        )))],
    ));

    d.extend(arithmetic());

    // Operator codes: 0 +, 1 -, 2 *, 3 /, 4 //, 5 %, 6 **, 7 &, 8 |,
    // 9 ^, 10 <<, 11 >>.
    let code = local("code", i64());
    d.push(define(
        "zb_any_arith",
        &[&code, &a, &b],
        any(),
        vec![
            ca.decl(category(a.e())),
            cb.decl(category(b.e())),
            when(
                and(and(is(&ca, STR), is(&cb, STR)), eq(code.e(), int(0))),
                vec![ret(box_str(add(get_str(a.e()), get_str(b.e()))))],
            ),
            when(
                and(and(is(&ca, STR), eq(code.e(), int(2))), is_integral(&cb)),
                vec![ret(box_str(call(
                    "zb_str_repeat",
                    vec![get_str(a.e()), number_i64(b.e(), cb.e())],
                    string(),
                )))],
            ),
            when(
                and(and(is(&cb, STR), eq(code.e(), int(2))), is_integral(&ca)),
                vec![ret(box_str(call(
                    "zb_str_repeat",
                    vec![get_str(b.e()), number_i64(a.e(), ca.e())],
                    string(),
                )))],
            ),
            when(
                and(and(is(&ca, CUSTOM), is(&cb, CUSTOM)), eq(code.e(), int(0))),
                vec![
                    when(
                        and(is_tuple(a.e()), is_tuple(b.e())),
                        vec![ret(call(
                            "zb_box_tuple",
                            vec![call(
                                "zb_list_concat_any",
                                vec![
                                    call("zb_unbox_tuple", vec![a.e()], anys.clone()),
                                    call("zb_unbox_tuple", vec![b.e()], anys.clone()),
                                ],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                    ret(call(
                        "zb_list_box_any",
                        vec![call(
                            "zb_list_concat_any",
                            vec![iter(a.e()), iter(b.e())],
                            anys.clone(),
                        )],
                        any(),
                    )),
                ],
            ),
            when(
                and(is_number(ca.e()), is_number(cb.e())),
                vec![
                    when(
                        or(is(&ca, FLOAT), is(&cb, FLOAT)),
                        vec![ret(call(
                            "zb_arith_f64",
                            vec![
                                code.e(),
                                number_f64(a.e(), ca.e()),
                                number_f64(b.e(), cb.e()),
                            ],
                            any(),
                        ))],
                    ),
                    ret(call(
                        "zb_arith_i64",
                        vec![
                            code.e(),
                            number_i64(a.e(), ca.e()),
                            number_i64(b.e(), cb.e()),
                        ],
                        any(),
                    )),
                ],
            ),
            type_error(add(
                add(
                    add(
                        text("unsupported operand type(s): "),
                        quoted(type_name(a.e())),
                    ),
                    text(" and "),
                ),
                quoted(type_name(b.e())),
            )),
            ret(a.e()),
        ],
    ));

    // Sequences behind a box.
    let unbox = |k: Kind, x: Expr| {
        call(
            &format!("zb_unbox_list_raw_{}", k.suffix()),
            vec![x],
            list_of(list_type, k.ty()),
        )
    };
    let to_any = |k: Kind, xs: Expr| {
        call(
            &format!("zb_list_to_any_{}", k.suffix()),
            vec![xs],
            anys.clone(),
        )
    };
    let kind_is = |k: Kind, x: Expr| eq(kind(x), int(k.list_tag() >> 8));
    d.push(define(
        "zb_any_iter",
        &[&x],
        anys.clone(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(to_any(
                    Kind::Str,
                    call(
                        "zb_str_chars",
                        vec![get_str(x.e())],
                        list_of(list_type, string()),
                    ),
                ))],
            ),
            when(
                or(
                    ne(cat.e(), int(CUSTOM)),
                    or(eq(kind(x.e()), int(FUNC_TAG >> 8)), is_instance(x.e())),
                ),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object is not iterable"),
                ))],
            ),
            when(
                is_dict(x.e()),
                vec![ret(call(
                    "zb_dict_keys",
                    vec![raw_any(x.e())],
                    anys.clone(),
                ))],
            ),
            when(
                kind_is(Kind::Int, x.e()),
                vec![ret(to_any(Kind::Int, unbox(Kind::Int, x.e())))],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![ret(to_any(Kind::Float, unbox(Kind::Float, x.e())))],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![ret(to_any(Kind::Str, unbox(Kind::Str, x.e())))],
            ),
            ret(unbox(Kind::Any, x.e())),
        ],
    ));
    let repr_of = |k: Kind, x: Expr| {
        call(
            &format!("zb_list_repr_{}", k.suffix()),
            vec![unbox(k, x)],
            string(),
        )
    };
    d.push(define(
        "zb_seq_repr_any",
        &[&x],
        string(),
        vec![
            when(
                is_instance(x.e()),
                vec![ret(call("zb_hook_instance_str", vec![x.e()], string()))],
            ),
            when(
                eq(kind(x.e()), int(FUNC_TAG >> 8)),
                vec![ret(add(add(text("<"), text(names.function)), text(">")))],
            ),
            when(
                kind_is(Kind::Int, x.e()),
                vec![ret(repr_of(Kind::Int, x.e()))],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![ret(repr_of(Kind::Float, x.e()))],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![ret(repr_of(Kind::Str, x.e()))],
            ),
            when(
                is_tuple(x.e()),
                vec![ret(call(
                    "zb_tuple_repr",
                    vec![call("zb_unbox_tuple", vec![x.e()], anys.clone())],
                    string(),
                ))],
            ),
            when(
                is_dict(x.e()),
                vec![ret(call("zb_dict_repr", vec![raw_any(x.e())], string()))],
            ),
            when(
                is_set(x.e()),
                vec![ret(call("zb_set_repr", vec![raw_any(x.e())], string()))],
            ),
            ret(repr_of(Kind::Any, x.e())),
        ],
    ));
    let len_of = |k: Kind, x: Expr| mcall(unbox(k, x), "len", vec![], i64());
    d.push(define(
        "zb_seq_len_any",
        &[&x],
        i64(),
        vec![
            when(
                kind_is(Kind::Int, x.e()),
                vec![ret(len_of(Kind::Int, x.e()))],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![ret(len_of(Kind::Float, x.e()))],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![ret(len_of(Kind::Str, x.e()))],
            ),
            when(
                is_dict(x.e()),
                vec![ret(call("zb_dict_len", vec![raw_any(x.e())], i64()))],
            ),
            ret(len_of(Kind::Any, x.e())),
        ],
    ));
    let i = local("i", any());
    let index = |i: Expr| call("zb_any_int", vec![i], i64());
    d.push(define(
        "zb_any_getitem",
        &[&x, &i],
        any(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(box_str(call(
                    "zb_str_get",
                    vec![get_str(x.e()), index(i.e())],
                    string(),
                )))],
            ),
            when(
                ne(cat.e(), int(CUSTOM)),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object is not subscriptable"),
                ))],
            ),
            when(
                is_dict(x.e()),
                vec![ret(call("zb_dict_get", vec![raw_any(x.e()), i.e()], any()))],
            ),
            ret(call(
                "zb_list_get_any",
                vec![iter(x.e()), index(i.e())],
                any(),
            )),
        ],
    ));
    let mutable_list = |x: Expr| and(eq(category(x.clone()), int(CUSTOM)), kind_is(Kind::Any, x));
    let boxed_dict = |x: Expr| and(eq(category(x.clone()), int(CUSTOM)), is_dict(x));
    let v = owned("v", any());
    d.push(define(
        "zb_any_setitem",
        &[&x, &i, &v],
        unit(),
        vec![
            when(
                boxed_dict(x.e()),
                vec![
                    expr(call(
                        "zb_dict_set",
                        vec![raw_any(x.e()), i.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                not(mutable_list(x.e())),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object does not support item assignment"),
                ))],
            ),
            expr(call(
                "zb_list_set_any",
                vec![unbox(Kind::Any, x.e()), index(i.e()), v.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    d.push(define(
        "zb_any_delitem",
        &[&x, &i],
        unit(),
        vec![
            when(
                boxed_dict(x.e()),
                vec![
                    expr(call("zb_dict_del", vec![raw_any(x.e()), i.e()], unit())),
                    ret_void(),
                ],
            ),
            when(
                not(mutable_list(x.e())),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object does not support item deletion"),
                ))],
            ),
            expr(call(
                "zb_list_pop_any",
                vec![unbox(Kind::Any, x.e()), index(i.e())],
                any(),
            )),
            ret_void(),
        ],
    ));
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let mask = local("mask", i64());
    let slice_any = |xs: Expr| {
        call(
            "zb_list_slice_any",
            vec![xs, start.e(), stop.e(), step.e(), mask.e()],
            anys.clone(),
        )
    };
    d.push(define(
        "zb_any_getslice",
        &[&x, &start, &stop, &step, &mask],
        any(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(box_str(call(
                    "zb_str_slice",
                    vec![get_str(x.e()), start.e(), stop.e(), step.e(), mask.e()],
                    string(),
                )))],
            ),
            when(
                ne(cat.e(), int(CUSTOM)),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object is not subscriptable"),
                ))],
            ),
            when(
                is_tuple(x.e()),
                vec![ret(call(
                    "zb_box_tuple",
                    vec![slice_any(call("zb_unbox_tuple", vec![x.e()], anys.clone()))],
                    any(),
                ))],
            ),
            ret(call("zb_list_box_any", vec![slice_any(iter(x.e()))], any())),
        ],
    ));

    d.extend(rounding());
    d
}

/// Floor division, remainder with the divisor's sign, integer powers,
/// and the operator-code dispatch over typed operands.
fn arithmetic() -> Vec<Decl> {
    let a = local("a", i64());
    let b = local("b", i64());
    let q = local("q", i64());
    let r = local("r", i64());
    let fa = local("a", f64());
    let fb = local("b", f64());
    let fr = local("r", f64());
    let code = local("code", i64());
    let mut d = Vec::new();

    let sign_differs = |r: Expr, b: Expr, zero: Expr| {
        and(
            ne(r.clone(), zero.clone()),
            ne(lt(r, zero.clone()), lt(b, zero)),
        )
    };
    d.push(define(
        "zb_floordiv_i64",
        &[&a, &b],
        i64(),
        vec![
            q.decl(div(a.e(), b.e())),
            r.decl(rem(a.e(), b.e())),
            when(
                sign_differs(r.e(), b.e(), int(0)),
                vec![ret(sub(q.e(), int(1)))],
            ),
            ret(q.e()),
        ],
    ));
    d.push(define(
        "zb_mod_i64",
        &[&a, &b],
        i64(),
        vec![
            r.decl(rem(a.e(), b.e())),
            when(
                sign_differs(r.e(), b.e(), int(0)),
                vec![ret(add(r.e(), b.e()))],
            ),
            ret(r.e()),
        ],
    ));
    d.push(define(
        "zb_mod_f64",
        &[&fa, &fb],
        f64(),
        vec![
            fr.decl(rem(fa.e(), fb.e())),
            when(
                sign_differs(fr.e(), fb.e(), float(0.0)),
                vec![ret(add(fr.e(), fb.e()))],
            ),
            ret(fr.e()),
        ],
    ));
    let result = local("result", i64());
    let base = local("base", i64());
    let exp = local("exp", i64());
    d.push(define(
        "zb_pow_i64",
        &[&a, &b],
        i64(),
        vec![
            result.decl(int(1)),
            base.decl(a.e()),
            exp.decl(b.e()),
            while_(
                gt(exp.e(), int(0)),
                vec![
                    when(
                        eq(bitand(exp.e(), int(1)), int(1)),
                        vec![result.set(mul(result.e(), base.e()))],
                    ),
                    base.set(mul(base.e(), base.e())),
                    exp.set(shr(exp.e(), int(1))),
                ],
            ),
            ret(result.e()),
        ],
    ));

    let box_i64 = |v: Expr| call("zb_box_i64", vec![v], any());
    let box_f64 = |v: Expr| call("zb_box_f64", vec![v], any());
    let is_code = |c: i64| eq(code.e(), int(c));
    let mut int_ops: Vec<Stmt> = vec![
        when(is_code(0), vec![ret(box_i64(add(a.e(), b.e())))]),
        when(is_code(1), vec![ret(box_i64(sub(a.e(), b.e())))]),
        when(is_code(2), vec![ret(box_i64(mul(a.e(), b.e())))]),
        when(
            is_code(3),
            vec![ret(box_f64(div(cast(a.e(), f64()), cast(b.e(), f64()))))],
        ),
        when(
            is_code(4),
            vec![ret(box_i64(call(
                "zb_floordiv_i64",
                vec![a.e(), b.e()],
                i64(),
            )))],
        ),
        when(
            is_code(5),
            vec![ret(box_i64(call("zb_mod_i64", vec![a.e(), b.e()], i64())))],
        ),
        when(
            is_code(6),
            vec![
                when(
                    lt(b.e(), int(0)),
                    vec![ret(box_f64(call(
                        "pow",
                        vec![cast(a.e(), f64()), cast(b.e(), f64())],
                        f64(),
                    )))],
                ),
                ret(box_i64(call("zb_pow_i64", vec![a.e(), b.e()], i64()))),
            ],
        ),
        when(is_code(7), vec![ret(box_i64(bitand(a.e(), b.e())))]),
        when(is_code(8), vec![ret(box_i64(bitor(a.e(), b.e())))]),
        when(is_code(9), vec![ret(box_i64(bitxor(a.e(), b.e())))]),
        when(is_code(10), vec![ret(box_i64(shl(a.e(), b.e())))]),
    ];
    int_ops.push(ret(box_i64(shr(a.e(), b.e()))));
    d.push(define("zb_arith_i64", &[&code, &a, &b], any(), int_ops));

    d.push(define(
        "zb_arith_f64",
        &[&code, &fa, &fb],
        any(),
        vec![
            when(is_code(0), vec![ret(box_f64(add(fa.e(), fb.e())))]),
            when(is_code(1), vec![ret(box_f64(sub(fa.e(), fb.e())))]),
            when(is_code(2), vec![ret(box_f64(mul(fa.e(), fb.e())))]),
            when(is_code(3), vec![ret(box_f64(div(fa.e(), fb.e())))]),
            when(
                is_code(4),
                vec![ret(box_f64(call(
                    "floor",
                    vec![div(fa.e(), fb.e())],
                    f64(),
                )))],
            ),
            when(
                is_code(5),
                vec![ret(box_f64(call(
                    "zb_mod_f64",
                    vec![fa.e(), fb.e()],
                    f64(),
                )))],
            ),
            ret(box_f64(call("pow", vec![fa.e(), fb.e()], f64()))),
        ],
    ));
    d
}

/// Rounding half to even, to an integer or to a number of digits.
fn rounding() -> Vec<Decl> {
    let x = local("x", f64());
    let whole = local("whole", f64());
    let diff = local("diff", f64());
    let r = local("r", f64());
    let half = local("half", i64());
    let digits = local("digits", i64());
    let scale = local("scale", f64());
    let scaled = local("scaled", f64());
    let round_up = |value: &Local| {
        vec![
            whole.decl(call("floor", vec![value.e()], f64())),
            diff.decl(sub(value.e(), whole.e())),
            r.decl(whole.e()),
            if_(
                gt(diff.e(), float(0.5)),
                vec![r.set(add(whole.e(), float(1.0)))],
                vec![when(
                    eq(diff.e(), float(0.5)),
                    vec![
                        half.decl(cast(whole.e(), i64())),
                        when(
                            ne(rem(half.e(), int(2)), int(0)),
                            vec![r.set(add(whole.e(), float(1.0)))],
                        ),
                    ],
                )],
            ),
        ]
    };
    let mut d = Vec::new();
    let mut body = round_up(&x);
    body.push(ret(cast(r.e(), i64())));
    d.push(define("zb_round_half_even", &[&x], i64(), body));
    let mut body = vec![
        scale.decl(call(
            "pow",
            vec![float(10.0), cast(digits.e(), f64())],
            f64(),
        )),
        scaled.decl(mul(x.e(), scale.e())),
    ];
    body.extend(round_up(&scaled));
    body.push(ret(div(r.e(), scale.e())));
    d.push(define("zb_round_digits", &[&x, &digits], f64(), body));
    let v = local("x", any());
    d.push(define(
        "zb_any_round",
        &[&v],
        any(),
        vec![
            when(
                eq(call("zb_any_category", vec![v.e()], i64()), int(FLOAT)),
                vec![ret(call(
                    "zb_box_i64",
                    vec![call(
                        "zb_round_half_even",
                        vec![call("zb_box_get_f64", vec![v.e()], f64())],
                        i64(),
                    )],
                    any(),
                ))],
            ),
            ret(v.e()),
        ],
    ));
    d
}
