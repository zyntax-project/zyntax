//! Dynamic values. An `Any` is a box with a type tag; the low byte of
//! the tag is the category (0 none, 1 bool, 2 signed int, 3 unsigned
//! int, 4 float, 5 string, 255 custom) and, for a boxed list or tuple,
//! the kind sits in the byte above. Every operation switches on the
//! category and then does what the typed code does.

use crate::build::*;
use crate::bytes::BYTES;
use crate::{DICT_TAG, FUNC_TAG, INSTANCE_KIND_BASE, Kind, Policy, SET_TAG, TUPLE_TAG, list_of};
use zyntax_typed_ast::TypeId;

pub(crate) const NONE: i64 = 0;
const BOOL: i64 = 1;
const INT: i64 = 2;
const UINT: i64 = 3;
const FLOAT: i64 = 4;
pub(crate) const STR: i64 = 5;
const CUSTOM: i64 = 255;
/// The whole tag of a box of the width its category's readers assume:
/// the width in the byte above the category.
pub(crate) const I64_TAG: i64 = (4 << 8) | INT;
pub(crate) const F64_TAG: i64 = (4 << 8) | FLOAT;

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
fn box_bytes(v: Expr) -> Expr {
    call("zb_box_bytes", vec![v], any())
}
fn get_i64(x: Expr) -> Expr {
    call("zb_box_get_i64", vec![x], i64())
}
fn get_f64(x: Expr) -> Expr {
    call("zb_box_get_f64", vec![x], f64())
}
/// The payload of a box whose tag has been read: the width is known,
/// and the read is two loads.
fn payload_i64(x: Expr) -> Expr {
    call("zb_box_payload_i64", vec![x], i64())
}
fn payload_f64(x: Expr) -> Expr {
    call("zb_box_payload_f64", vec![x], f64())
}
fn payload_bool(x: Expr) -> Expr {
    call("zb_box_payload_bool", vec![x], i32())
}
/// A bool box's value; the box has been checked to be one.
fn get_bool(x: Expr) -> Expr {
    ne(payload_bool(x), int32(0))
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
    and(
        ge(kind(x.clone()), int(INSTANCE_KIND_BASE)),
        lt(kind(x), int(crate::lists::SHAPE_KIND_BASE)),
    )
}

/// A boxed list whose elements are a tuple shape the frontend
/// registered; its elements are reached through the frontend's hooks.
fn is_shaped(x: Expr) -> Expr {
    and(
        eq(category(x.clone()), int(CUSTOM)),
        ge(kind(x), int(crate::lists::SHAPE_KIND_BASE)),
    )
}

fn shaped_items(x: Expr, anys: zyntax_typed_ast::Type) -> Expr {
    call("zb_hook_shaped_items", vec![x], anys)
}

/// The instance hooks as the frontend that defines them declares them.
pub(crate) fn extern_instance_hooks() -> Vec<Decl> {
    vec![
        extern_fn("zb_hook_instance_str", &[("x", any())], string(), None),
        extern_fn("zb_hook_instance_type", &[("x", any())], string(), None),
        extern_fn(
            "zb_hook_instance_eq",
            &[("a", any()), ("b", any())],
            boolean(),
            None,
        ),
        extern_fn("zb_hook_instance_hash", &[("x", any())], i64(), None),
        extern_fn(
            "zb_hook_instance_arith",
            &[("code", i64()), ("a", any()), ("b", any())],
            any(),
            None,
        ),
    ]
}

/// What the library says about instances when the frontend defines no
/// hooks: nothing is equal to anything, and they print as objects.
pub(crate) fn default_instance_hooks(policy: &Policy) -> Vec<Decl> {
    let x = local("x", any());
    let a = local("a", any());
    let b = local("b", any());
    let code = local("code", i64());
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
        define(
            "zb_hook_instance_hash",
            &[&x],
            i64(),
            vec![ret(call("zb_unbox_instance_raw", vec![x.e()], i64()))],
        ),
        define(
            "zb_hook_instance_arith",
            &[&code, &a, &b],
            any(),
            vec![
                fatal(
                    "TypeError",
                    text("unsupported operand type(s) for an object"),
                ),
                ret(null(any())),
            ],
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
        Some("zyntax_box_pointer"),
    ));
    // A null instance is None, and None is a null box.
    let p = local("p", i64());
    let itag = local("tag", i32());
    d.push(define(
        "zb_box_instance",
        &[&p, &itag],
        any(),
        vec![
            when(eq(p.e(), int(0)), vec![ret(null(any()))]),
            ret(call("zb_box_instance_raw", vec![p.e(), itag.e()], any())),
        ],
    ));
    d.push(define(
        "zb_unbox_instance",
        &[&x],
        i64(),
        vec![
            when(eq(x.e(), null(any())), vec![ret(int(0))]),
            ret(call("zb_unbox_instance_raw", vec![x.e()], i64())),
        ],
    ));
    d.push(extern_fn(
        "zb_box_free",
        &[("x", any())],
        unit(),
        Some("zyntax_box_free"),
    ));
    // A caught exception the handler is done with: the instance, then
    // the box it travelled in. What the instance's fields hold stays.
    d.push(extern_fn("free", &[("p", string())], unit(), None));
    d.push(define(
        "zb_release_caught",
        &[&x],
        unit(),
        vec![
            when(eq(x.e(), null(any())), vec![ret_void()]),
            expr(call(
                "free",
                vec![cast(
                    call("zb_unbox_instance_raw", vec![x.e()], i64()),
                    string(),
                )],
                unit(),
            )),
            expr(call("zb_box_free", vec![x.e()], unit())),
            ret_void(),
        ],
    ));
    // The box accessors. The `data` and payload readers take a box that
    // is known to be one, and are the loads they name; the tag reader
    // answers None with the void tag first.
    d.push(extern_fn(
        "zb_box_header_tag",
        &[("x", any())],
        i32(),
        Some("zyntax_box_header_tag"),
    ));
    d.push(define(
        "zb_box_tag",
        &[&x],
        i32(),
        vec![
            when(eq(x.e(), null(any())), vec![ret(int32(0))]),
            ret(call("zb_box_header_tag", vec![x.e()], i32())),
        ],
    ));
    for (name, ty, link) in [
        ("zb_box_payload_i64", i64(), "zyntax_box_payload_i64"),
        ("zb_box_payload_f64", f64(), "zyntax_box_payload_f64"),
        ("zb_box_payload_bool", i32(), "zyntax_box_payload_bool"),
        ("zb_box_hash", i64(), "zyntax_box_hash"),
    ] {
        d.push(extern_fn(name, &[("x", any())], ty, Some(link)));
    }
    d.push(extern_fn(
        "zb_box_set_hash",
        &[("x", any()), ("h", i64())],
        unit(),
        Some("zyntax_box_set_hash"),
    ));
    d.push(define(
        "zb_box_payload_truth",
        &[&x],
        boolean(),
        vec![ret(get_bool(x.e()))],
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
        Some("zyntax_box_data"),
    ));

    // A typed value becomes a dynamic one by being returned as one. A
    // string's box holds its own copy and frees it with the box, so the
    // string given stays the caller's.
    for (name, ty) in [
        ("zb_box_i64", i64()),
        ("zb_box_f64", f64()),
        ("zb_box_bool", boolean()),
    ] {
        let v = local("v", ty);
        d.push(define(name, &[&v], any(), vec![ret(v.e())]));
    }
    d.push(extern_fn(
        "zb_str_to_dynamic",
        &[("s", string())],
        any(),
        Some("$IO$string_to_dynamic"),
    ));
    let s_in = local("v", string());
    d.push(define(
        "zb_box_str",
        &[&s_in],
        any(),
        vec![ret(call("zb_str_to_dynamic", vec![s_in.e()], any()))],
    ));

    d.push(define(
        "zb_any_category",
        &[&x],
        i64(),
        vec![ret(bitand(tag(x.e()), int(255)))],
    ));
    // Whether the box holds an instance of a frontend class.
    d.push(define(
        "zb_any_is_instance",
        &[&x],
        boolean(),
        vec![ret(and(
            eq(category(x.e()), int(CUSTOM)),
            is_instance(x.e()),
        ))],
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
    // A number as f64 or i64, given its category. The boxes this library
    // and its frontends make are 64 bits wide or a bool byte, read as
    // loads; a box of another width comes from a plugin and goes through
    // the runtime's reader.
    d.push(define(
        "zb_number_f64",
        &[&x, &cat],
        f64(),
        vec![
            when(eq(tag(x.e()), int(F64_TAG)), vec![ret(payload_f64(x.e()))]),
            when(is(&cat, FLOAT), vec![ret(get_f64(x.e()))]),
            when(
                is(&cat, BOOL),
                vec![
                    when(get_bool(x.e()), vec![ret(float(1.0))]),
                    ret(float(0.0)),
                ],
            ),
            when(
                eq(tag(x.e()), int(I64_TAG)),
                vec![ret(cast(payload_i64(x.e()), f64()))],
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
            when(eq(tag(x.e()), int(I64_TAG)), vec![ret(payload_i64(x.e()))]),
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
            when(is(&cat, BYTES), vec![ret(text(names.bytes))]),
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
                is(&cat, BYTES),
                vec![ret(ne(
                    call("zb_str_len", vec![get_str(x.e())], i64()),
                    int(0),
                ))],
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
                is(&cat, BYTES),
                vec![ret(call("zb_bytes_repr", vec![get_str(x.e())], string()))],
            ),
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
                or(is(&ca, BYTES), is(&cb, BYTES)),
                vec![
                    when(
                        and(is(&ca, BYTES), is(&cb, BYTES)),
                        vec![ret(ne(
                            call("zb_bytes_eq", vec![get_str(a.e()), get_str(b.e())], i32()),
                            int32(0),
                        ))],
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
                    // Two tuples, or two lists of dynamic values, are
                    // their storage already.
                    when(
                        and(
                            or(
                                is_tuple(a.e()),
                                eq(kind(a.e()), int(Kind::Any.list_tag() >> 8)),
                            ),
                            or(
                                is_tuple(b.e()),
                                eq(kind(b.e()), int(Kind::Any.list_tag() >> 8)),
                            ),
                        ),
                        vec![ret(call(
                            "zb_list_eq_any",
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
    // The kind a sort key is compared as when every key shares it: 1
    // for an int, 2 for a float, 3 for a string, 0 for anything else.
    d.push(define(
        "zb_any_key_kind",
        &[&x],
        i64(),
        vec![
            cat.decl(category(x.e())),
            when(is_int(&cat), vec![ret(int(1))]),
            when(is(&cat, FLOAT), vec![ret(int(2))]),
            when(is(&cat, STR), vec![ret(int(3))]),
            ret(int(0)),
        ],
    ));

    // What a dict keys a value by: equal values hash equal, so a
    // number hashes as its integral value when it has one, a string by
    // its bytes, a tuple by its elements, an instance through its
    // class, and None as itself. A list, dict or set cannot be a key.
    let h = local("h", i64());
    let items = local("items", anys.clone());
    let elem = local("elem", any());
    let n = local("n", i64());
    let i = local("i", i64());
    // The hash of a float: an integral value hashes as the integer it
    // is, so it meets that integer in a table; anything else by its
    // scaled bits.
    let fv = local("f", f64());
    d.push(define(
        "zb_hash_of_f64",
        &[&fv],
        i64(),
        vec![
            when(
                and(
                    eq(fv.e(), cast(cast(fv.e(), i64()), f64())),
                    and(gt(fv.e(), float(-9.2e18)), lt(fv.e(), float(9.2e18))),
                ),
                vec![ret(cast(fv.e(), i64()))],
            ),
            ret(cast(mul(fv.e(), float(1_048_576.0)), i64())),
        ],
    ));
    // The hash of a string, never zero, which is what a string box
    // records as "not yet hashed".
    let sv = local("s", string());
    d.push(define(
        "zb_hash_of_str",
        &[&sv],
        i64(),
        vec![
            h.decl(call("zb_str_hash", vec![sv.e()], i64())),
            when(eq(h.e(), int(0)), vec![ret(int(1))]),
            ret(h.e()),
        ],
    ));
    d.push(define(
        "zb_any_hash",
        &[&x],
        i64(),
        vec![
            cat.decl(category(x.e())),
            when(is(&cat, NONE), vec![ret(int(0x5A6E_6F6E_6521))]),
            when(is_integral(&cat), vec![ret(number_i64(x.e(), cat.e()))]),
            when(
                is(&cat, FLOAT),
                vec![ret(call("zb_hash_of_f64", vec![get_f64(x.e())], i64()))],
            ),
            // A string box keeps its hash once computed, zero until then.
            when(
                is(&cat, STR),
                vec![
                    h.decl(call("zb_box_hash", vec![x.e()], i64())),
                    when(ne(h.e(), int(0)), vec![ret(h.e())]),
                    h.set(call("zb_hash_of_str", vec![get_str(x.e())], i64())),
                    expr(call("zb_box_set_hash", vec![x.e(), h.e()], unit())),
                    ret(h.e()),
                ],
            ),
            when(
                is(&cat, BYTES),
                vec![ret(call("zb_bytes_hash", vec![get_str(x.e())], i64()))],
            ),
            when(
                is_instance(x.e()),
                vec![ret(call("zb_hook_instance_hash", vec![x.e()], i64()))],
            ),
            when(
                is_tuple(x.e()),
                vec![
                    items.decl(call("zb_unbox_tuple", vec![x.e()], anys.clone())),
                    n.decl(mcall(items.e(), "len", vec![], i64())),
                    h.decl(int(0x2545_F491_4F6C_DD1D)),
                    i.decl(int(0)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            elem.decl(idx(items.e(), i.e(), any())),
                            // An integer element hashes to itself, read
                            // without the dispatch.
                            if_(
                                eq(tag(elem.e()), int(I64_TAG)),
                                vec![h.set(add(mul(h.e(), int(1_000_003)), payload_i64(elem.e())))],
                                vec![h.set(add(
                                    mul(h.e(), int(1_000_003)),
                                    call("zb_any_hash", vec![elem.e()], i64()),
                                ))],
                            ),
                            i.add_assign(int(1)),
                        ],
                    ),
                    ret(h.e()),
                ],
            ),
            when(
                eq(kind(x.e()), int(FUNC_TAG >> 8)),
                vec![ret(call("zb_unbox_instance_raw", vec![x.e()], i64()))],
            ),
            // A frozenset hashes by its contents; a set is not told
            // apart from one.
            when(
                is_set(x.e()),
                vec![ret(call("zb_set_hash", vec![raw_any(x.e())], i64()))],
            ),
            type_error(add(text("unhashable type: "), quoted(type_name(x.e())))),
            ret(int(0)),
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
                and(is_set(a.e()), is_set(b.e())),
                vec![ret(and(
                    call(
                        "zb_set_issubset",
                        vec![raw_any(a.e()), raw_any(b.e())],
                        boolean(),
                    ),
                    not(call(
                        "zb_set_eq",
                        vec![raw_any(a.e()), raw_any(b.e())],
                        boolean(),
                    )),
                ))],
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
    d.push(define(
        "zb_any_le",
        &[&a, &b],
        boolean(),
        vec![
            when(
                and(is_set(a.e()), is_set(b.e())),
                vec![ret(call(
                    "zb_set_issubset",
                    vec![raw_any(a.e()), raw_any(b.e())],
                    boolean(),
                ))],
            ),
            ret(not(call("zb_any_lt", vec![b.e(), a.e()], boolean()))),
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
                is_set(container.e()),
                vec![ret(call(
                    "zb_set_contains",
                    vec![raw_any(container.e()), item.e()],
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
                is(&cat, BYTES),
                vec![ret(call("zb_str_len", vec![get_str(x.e())], i64()))],
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

    // A value read back as the type it was declared to be. Unlike the
    // conversions below these change nothing: a box of another type
    // is a TypeError.
    d.push(define(
        "zb_any_as_i64",
        &[&x],
        i64(),
        vec![
            cat.decl(category(x.e())),
            when(
                not(is_integral(&cat)),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object cannot be interpreted as an integer"),
                ))],
            ),
            ret(number_i64(x.e(), cat.e())),
        ],
    ));
    d.push(define(
        "zb_any_as_f64",
        &[&x],
        f64(),
        vec![
            cat.decl(category(x.e())),
            when(
                not(is_number(cat.e())),
                vec![type_error(add(
                    text("must be real number, not "),
                    type_name(x.e()),
                ))],
            ),
            ret(number_f64(x.e(), cat.e())),
        ],
    ));
    d.push(define(
        "zb_any_as_str",
        &[&x],
        string(),
        vec![
            cat.decl(category(x.e())),
            when(
                not(is(&cat, STR)),
                vec![type_error(add(
                    add(text("expected str instance, "), type_name(x.e())),
                    text(" found"),
                ))],
            ),
            ret(get_str(x.e())),
        ],
    ));
    d.push(define(
        "zb_any_as_bool",
        &[&x],
        boolean(),
        vec![
            cat.decl(category(x.e())),
            when(
                not(is_integral(&cat)),
                vec![type_error(add(
                    quoted(type_name(x.e())),
                    text(" object cannot be interpreted as an integer"),
                ))],
            ),
            ret(ne(number_i64(x.e(), cat.e()), int(0))),
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
    let same = kept("x", any());
    d.push(define("zb_any_pos", &[&same], any(), vec![ret(same.e())]));
    let f = local("f", f64());
    let n = local("n", i64());
    d.push(define(
        "zb_any_abs",
        &[&x],
        any(),
        vec![
            cat.decl(category(x.e())),
            when(
                is(&cat, FLOAT),
                vec![
                    f.decl(get_f64(x.e())),
                    when(lt(f.e(), float(0.0)), vec![f.set(sub(float(0.0), f.e()))]),
                    ret(box_f64(f.e())),
                ],
            ),
            n.decl(number_i64(x.e(), cat.e())),
            when(lt(n.e(), int(0)), vec![n.set(sub(int(0), n.e()))]),
            ret(box_i64(n.e())),
        ],
    ));
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
                and(and(is(&ca, BYTES), is(&cb, BYTES)), eq(code.e(), int(0))),
                vec![ret(box_bytes(add(get_str(a.e()), get_str(b.e()))))],
            ),
            when(
                and(and(is(&ca, BYTES), eq(code.e(), int(2))), is_integral(&cb)),
                vec![ret(box_bytes(call(
                    "zb_bytes_repeat",
                    vec![get_str(a.e()), number_i64(b.e(), cb.e())],
                    string(),
                )))],
            ),
            when(
                and(and(is(&cb, BYTES), eq(code.e(), int(2))), is_integral(&ca)),
                vec![ret(box_bytes(call(
                    "zb_bytes_repeat",
                    vec![get_str(b.e()), number_i64(a.e(), ca.e())],
                    string(),
                )))],
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
                and(is_set(a.e()), is_set(b.e())),
                vec![
                    when(
                        eq(code.e(), int(1)),
                        vec![ret(call(
                            "zb_set_box",
                            vec![call(
                                "zb_set_sub",
                                vec![raw_any(a.e()), raw_any(b.e())],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                    when(
                        eq(code.e(), int(7)),
                        vec![ret(call(
                            "zb_set_box",
                            vec![call(
                                "zb_set_and",
                                vec![raw_any(a.e()), raw_any(b.e())],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                    when(
                        eq(code.e(), int(8)),
                        vec![ret(call(
                            "zb_set_box",
                            vec![call(
                                "zb_set_or",
                                vec![raw_any(a.e()), raw_any(b.e())],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                    when(
                        eq(code.e(), int(9)),
                        vec![ret(call(
                            "zb_set_box",
                            vec![call(
                                "zb_set_xor",
                                vec![raw_any(a.e()), raw_any(b.e())],
                                anys.clone(),
                            )],
                            any(),
                        ))],
                    ),
                ],
            ),
            // An instance on either side takes part through its class.
            when(
                or(
                    and(is(&ca, CUSTOM), is_instance(a.e())),
                    and(is(&cb, CUSTOM), is_instance(b.e())),
                ),
                vec![ret(call(
                    "zb_hook_instance_arith",
                    vec![code.e(), a.e(), b.e()],
                    any(),
                ))],
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
            ret(null(any())),
        ],
    ));
    // The same with an integer on one side, which is then never boxed:
    // a number on the other side computes directly, anything else goes
    // the general way.
    let m = local("m", i64());
    d.push(define(
        "zb_any_arith_i64",
        &[&code, &a, &m],
        any(),
        vec![
            ca.decl(category(a.e())),
            when(
                is_number(ca.e()),
                vec![
                    when(
                        is(&ca, FLOAT),
                        vec![ret(call(
                            "zb_arith_f64",
                            vec![code.e(), get_f64(a.e()), cast(m.e(), f64())],
                            any(),
                        ))],
                    ),
                    ret(call(
                        "zb_arith_i64",
                        vec![code.e(), number_i64(a.e(), ca.e()), m.e()],
                        any(),
                    )),
                ],
            ),
            ret(call(
                "zb_any_arith",
                vec![code.e(), a.e(), box_i64(m.e())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zb_i64_arith_any",
        &[&code, &m, &b],
        any(),
        vec![
            cb.decl(category(b.e())),
            when(
                is_number(cb.e()),
                vec![
                    when(
                        is(&cb, FLOAT),
                        vec![ret(call(
                            "zb_arith_f64",
                            vec![code.e(), cast(m.e(), f64()), get_f64(b.e())],
                            any(),
                        ))],
                    ),
                    ret(call(
                        "zb_arith_i64",
                        vec![code.e(), m.e(), number_i64(b.e(), cb.e())],
                        any(),
                    )),
                ],
            ),
            ret(call(
                "zb_any_arith",
                vec![code.e(), box_i64(m.e()), b.e()],
                any(),
            )),
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
            when(
                eq(tag(x.e()), int(TUPLE_TAG)),
                vec![ret(call("zb_unbox_tuple_raw", vec![x.e()], anys.clone()))],
            ),
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
                is(&cat, BYTES),
                vec![ret(to_any(
                    Kind::Int,
                    call(
                        "zb_bytes_to_list",
                        vec![get_str(x.e())],
                        list_of(list_type, i64()),
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
                is_set(x.e()),
                vec![ret(call(
                    "zb_set_items",
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
            when(
                kind_is(Kind::Ptr, x.e()),
                vec![ret(to_any(Kind::Ptr, unbox(Kind::Ptr, x.e())))],
            ),
            when(
                is_shaped(x.e()),
                vec![ret(shaped_items(x.e(), anys.clone()))],
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
                kind_is(Kind::Ptr, x.e()),
                vec![ret(repr_of(Kind::Ptr, x.e()))],
            ),
            when(
                is_shaped(x.e()),
                vec![ret(call("zb_hook_shaped_repr", vec![x.e()], string()))],
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
                kind_is(Kind::Ptr, x.e()),
                vec![ret(len_of(Kind::Ptr, x.e()))],
            ),
            when(
                is_dict(x.e()),
                vec![ret(call("zb_dict_len", vec![raw_any(x.e())], i64()))],
            ),
            when(
                is_set(x.e()),
                vec![ret(call("zb_set_len", vec![raw_any(x.e())], i64()))],
            ),
            ret(len_of(Kind::Any, x.e())),
        ],
    ));
    let i = local("i", any());
    let index = |i: Expr| call("zb_any_int", vec![i], i64());
    let list_get = |k: Kind, x: Expr, i: Expr| {
        call(
            &format!("zb_list_get_{}", k.suffix()),
            vec![unbox(k, x), index(i)],
            k.ty(),
        )
    };
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
                is(&cat, BYTES),
                vec![ret(box_i64(call(
                    "zb_bytes_index",
                    vec![get_str(x.e()), index(i.e())],
                    i64(),
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
            // A list of one kind is read in place, the element boxed.
            when(
                kind_is(Kind::Int, x.e()),
                vec![ret(box_i64(list_get(Kind::Int, x.e(), i.e())))],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![ret(box_f64(list_get(Kind::Float, x.e(), i.e())))],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![ret(box_str(list_get(Kind::Str, x.e(), i.e())))],
            ),
            when(
                kind_is(Kind::Ptr, x.e()),
                vec![ret(call(
                    "zb_hook_box_instance",
                    vec![list_get(Kind::Ptr, x.e(), i.e())],
                    any(),
                ))],
            ),
            when(
                is_shaped(x.e()),
                vec![ret(call(
                    "zb_hook_shaped_get",
                    vec![x.e(), index(i.e())],
                    any(),
                ))],
            ),
            ret(call(
                "zb_list_get_any",
                vec![iter(x.e()), index(i.e())],
                any(),
            )),
        ],
    ));
    // Positional indexing already has an unboxed integer in the Python
    // frontend (notably while unpacking nested tuples). Keep that integer
    // through the dispatch instead of allocating a box only to read it
    // back in every list branch. Dicts still need a boxed key.
    let position = local("position", i64());
    d.push(define(
        "zb_any_getitem_i64",
        &[&x, &position],
        any(),
        vec![
            // Tuple unpacking can read its backing list directly.
            when(
                eq(tag(x.e()), int(TUPLE_TAG)),
                vec![ret(call(
                    "zb_list_get_any",
                    vec![
                        call("zb_unbox_tuple_raw", vec![x.e()], anys.clone()),
                        position.e(),
                    ],
                    any(),
                ))],
            ),
            cat.decl(category(x.e())),
            when(
                is(&cat, STR),
                vec![ret(box_str(call(
                    "zb_str_get",
                    vec![get_str(x.e()), position.e()],
                    string(),
                )))],
            ),
            when(
                is(&cat, BYTES),
                vec![ret(box_i64(call(
                    "zb_bytes_index",
                    vec![get_str(x.e()), position.e()],
                    i64(),
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
                vec![ret(call(
                    "zb_dict_get",
                    vec![raw_any(x.e()), box_i64(position.e())],
                    any(),
                ))],
            ),
            when(
                kind_is(Kind::Int, x.e()),
                vec![ret(box_i64(call(
                    "zb_list_get_i64",
                    vec![unbox(Kind::Int, x.e()), position.e()],
                    i64(),
                )))],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![ret(box_f64(call(
                    "zb_list_get_f64",
                    vec![unbox(Kind::Float, x.e()), position.e()],
                    f64(),
                )))],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![ret(box_str(call(
                    "zb_list_get_str",
                    vec![unbox(Kind::Str, x.e()), position.e()],
                    string(),
                )))],
            ),
            when(
                kind_is(Kind::Ptr, x.e()),
                vec![ret(call(
                    "zb_hook_box_instance",
                    vec![call(
                        "zb_list_get_ptr",
                        vec![unbox(Kind::Ptr, x.e()), position.e()],
                        Kind::Ptr.ty(),
                    )],
                    any(),
                ))],
            ),
            when(
                is_shaped(x.e()),
                vec![ret(call(
                    "zb_hook_shaped_get",
                    vec![x.e(), position.e()],
                    any(),
                ))],
            ),
            ret(call(
                "zb_list_get_any",
                vec![iter(x.e()), position.e()],
                any(),
            )),
        ],
    ));
    let mutable_list = |x: Expr| {
        and(
            eq(category(x.clone()), int(CUSTOM)),
            or(
                kind_is(Kind::Any, x.clone()),
                or(
                    kind_is(Kind::Int, x.clone()),
                    or(
                        kind_is(Kind::Float, x.clone()),
                        or(
                            kind_is(Kind::Str, x.clone()),
                            or(kind_is(Kind::Ptr, x.clone()), is_shaped(x)),
                        ),
                    ),
                ),
            ),
        )
    };
    // A value stored into a list of one kind must be of that kind.
    let list_set = |k: Kind, x: Expr, i: Expr, v: Expr| {
        expr(call(
            &format!("zb_list_set_{}", k.suffix()),
            vec![unbox(k, x), index(i), v],
            unit(),
        ))
    };
    let boxed_dict = |x: Expr| and(eq(category(x.clone()), int(CUSTOM)), is_dict(x));
    let v = kept("v", any());
    // A dict keeps the key it is stored under.
    let key = kept("i", any());
    d.push(define(
        "zb_any_setitem",
        &[&x, &key, &v],
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
            when(
                kind_is(Kind::Int, x.e()),
                vec![
                    list_set(
                        Kind::Int,
                        x.e(),
                        i.e(),
                        call("zb_any_as_i64", vec![v.e()], i64()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![
                    list_set(
                        Kind::Float,
                        x.e(),
                        i.e(),
                        call("zb_any_as_f64", vec![v.e()], f64()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![
                    list_set(
                        Kind::Str,
                        x.e(),
                        i.e(),
                        call("zb_any_as_str", vec![v.e()], string()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                is_shaped(x.e()),
                vec![
                    expr(call(
                        "zb_hook_shaped_set",
                        vec![x.e(), index(i.e()), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Ptr, x.e()),
                vec![
                    when(
                        not(is_instance(v.e())),
                        vec![type_error(add(
                            text("expected an instance, got "),
                            quoted(type_name(v.e())),
                        ))],
                    ),
                    list_set(
                        Kind::Ptr,
                        x.e(),
                        i.e(),
                        cast(call("zb_unbox_instance_raw", vec![v.e()], i64()), usize()),
                    ),
                    ret_void(),
                ],
            ),
            expr(call(
                "zb_list_set_any",
                vec![unbox(Kind::Any, x.e()), index(i.e()), v.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    let list_set_i64 = |k: Kind, x: Expr, i: Expr, v: Expr| {
        expr(call(
            &format!("zb_list_set_{}", k.suffix()),
            vec![unbox(k, x), i, v],
            unit(),
        ))
    };
    d.push(define(
        "zb_any_setitem_i64",
        &[&x, &position, &v],
        unit(),
        vec![
            when(
                boxed_dict(x.e()),
                vec![
                    expr(call(
                        "zb_dict_set",
                        vec![raw_any(x.e()), box_i64(position.e()), v.e()],
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
            when(
                kind_is(Kind::Int, x.e()),
                vec![
                    list_set_i64(
                        Kind::Int,
                        x.e(),
                        position.e(),
                        call("zb_any_as_i64", vec![v.e()], i64()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Float, x.e()),
                vec![
                    list_set_i64(
                        Kind::Float,
                        x.e(),
                        position.e(),
                        call("zb_any_as_f64", vec![v.e()], f64()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Str, x.e()),
                vec![
                    list_set_i64(
                        Kind::Str,
                        x.e(),
                        position.e(),
                        call("zb_any_as_str", vec![v.e()], string()),
                    ),
                    ret_void(),
                ],
            ),
            when(
                is_shaped(x.e()),
                vec![
                    expr(call(
                        "zb_hook_shaped_set",
                        vec![x.e(), position.e(), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                kind_is(Kind::Ptr, x.e()),
                vec![
                    when(
                        not(is_instance(v.e())),
                        vec![type_error(add(
                            text("expected an instance, got "),
                            quoted(type_name(v.e())),
                        ))],
                    ),
                    list_set_i64(
                        Kind::Ptr,
                        x.e(),
                        position.e(),
                        cast(call("zb_unbox_instance_raw", vec![v.e()], i64()), usize()),
                    ),
                    ret_void(),
                ],
            ),
            expr(call(
                "zb_list_set_any",
                vec![unbox(Kind::Any, x.e()), position.e(), v.e()],
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
                is(&cat, BYTES),
                vec![ret(box_bytes(call(
                    "zb_bytes_slice",
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
    // Round `value` to the nearest whole, ties to even. `err` is the
    // sign of what rounding `value` itself lost, when it came from a
    // computation: a half that was rounded up to is not a tie.
    let err = local("err", f64());
    let round_up = |value: &Local, err: Option<&Local>| {
        let tie = match err {
            None => vec![
                half.decl(cast(whole.e(), i64())),
                when(
                    ne(rem(half.e(), int(2)), int(0)),
                    vec![r.set(add(whole.e(), float(1.0)))],
                ),
            ],
            Some(err) => vec![if_(
                gt(err.e(), float(0.0)),
                vec![r.set(add(whole.e(), float(1.0)))],
                vec![when(
                    eq(err.e(), float(0.0)),
                    vec![
                        half.decl(cast(whole.e(), i64())),
                        when(
                            ne(rem(half.e(), int(2)), int(0)),
                            vec![r.set(add(whole.e(), float(1.0)))],
                        ),
                    ],
                )],
            )],
        };
        vec![
            whole.decl(call("floor", vec![value.e()], f64())),
            diff.decl(sub(value.e(), whole.e())),
            r.decl(whole.e()),
            if_(
                gt(diff.e(), float(0.5)),
                vec![r.set(add(whole.e(), float(1.0)))],
                vec![when(eq(diff.e(), float(0.5)), tie)],
            ),
        ]
    };
    let mut d = Vec::new();
    let mut body = round_up(&x, None);
    body.push(ret(cast(r.e(), i64())));
    d.push(define("zb_round_half_even", &[&x], i64(), body));
    // Scaling by a power of ten is inexact, so the tie test reads the
    // exact residual of the scaling through a fused multiply-add: a
    // scaled value that landed on a half from below is not a tie.
    let magnitude = local("magnitude", i64());
    let mut body = vec![
        magnitude.decl(if_expr(
            lt(digits.e(), int(0)),
            sub(int(0), digits.e()),
            digits.e(),
        )),
        scale.decl(call(
            "pow",
            vec![float(10.0), cast(magnitude.e(), f64())],
            f64(),
        )),
        scaled.decl(if_expr(
            lt(digits.e(), int(0)),
            div(x.e(), scale.e()),
            mul(x.e(), scale.e()),
        )),
        err.decl(if_expr(
            lt(digits.e(), int(0)),
            call(
                "fma",
                vec![scaled.e(), sub(float(0.0), scale.e()), x.e()],
                f64(),
            ),
            call(
                "fma",
                vec![x.e(), scale.e(), sub(float(0.0), scaled.e())],
                f64(),
            ),
        )),
    ];
    body.extend(round_up(&scaled, Some(&err)));
    body.push(ret(if_expr(
        lt(digits.e(), int(0)),
        mul(r.e(), scale.e()),
        div(r.e(), scale.e()),
    )));
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
    // With digits: a float stays a float; an int rounds only when the
    // digits are negative, and stays an int.
    let digits = local("digits", i64());
    d.push(define(
        "zb_any_round_digits",
        &[&v, &digits],
        any(),
        vec![
            when(
                eq(call("zb_any_category", vec![v.e()], i64()), int(FLOAT)),
                vec![ret(call(
                    "zb_box_f64",
                    vec![call(
                        "zb_round_digits",
                        vec![call("zb_box_get_f64", vec![v.e()], f64()), digits.e()],
                        f64(),
                    )],
                    any(),
                ))],
            ),
            when(ge(digits.e(), int(0)), vec![ret(v.e())]),
            ret(call(
                "zb_box_i64",
                vec![cast(
                    call(
                        "zb_round_digits",
                        vec![
                            cast(call("zb_any_int", vec![v.e()], i64()), f64()),
                            digits.e(),
                        ],
                        f64(),
                    ),
                    i64(),
                )],
                any(),
            )),
        ],
    ));
    d
}
