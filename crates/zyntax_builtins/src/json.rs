//! `json.dumps` over dynamic values: pieces gathered into one list and
//! joined once, so the text costs its length. Strings are escaped by
//! the host; numbers print as Python prints them.

use crate::build::*;
use crate::{DICT_TAG, TUPLE_TAG, list_of};
use zyntax_typed_ast::TypeId;

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let strs = list_of(list_type, string());
    let mut d = Vec::new();
    d.push(extern_fn(
        "zb_json_escape",
        &[("s", string())],
        string(),
        Some("$Host$json_escape"),
    ));

    let pieces = borrowed("pieces", strs.clone());
    let x = local("x", any());
    let items = local("items", anys.clone());
    let n = local("n", i64());
    let i = local("i", i64());
    let cat = local("cat", i64());
    let piece = |p: Expr| expr(mcall(pieces.e(), "push", vec![p], unit()));
    let category = |x: Expr| call("zb_any_category", vec![x], i64());
    let kind = |x: Expr| call("zb_any_kind", vec![x], i64());
    let into = |v: Expr| expr(call("zb_json_into", vec![pieces.e(), v], unit()));
    let raw = |x: Expr| call("zb_unbox_list_raw_any", vec![x], anys.clone());
    // The elements of a list or tuple, then each value of a dict after
    // its key, between the brackets and separators JSON spells.
    let sequence = |items_of: Expr| {
        vec![
            items.decl(items_of),
            piece(text("[")),
            n.decl(mcall(items.e(), "len", vec![], i64())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(gt(i.e(), int(0)), vec![piece(text(", "))]),
                    into(idx(items.e(), i.e(), any())),
                    i.add_assign(int(1)),
                ],
            ),
            piece(text("]")),
            ret_void(),
        ]
    };
    d.push(define(
        "zb_json_into",
        &[&pieces, &x],
        unit(),
        vec![
            cat.decl(category(x.e())),
            // None, bools and numbers.
            when(eq(cat.e(), int(0)), vec![piece(text("null")), ret_void()]),
            when(
                eq(cat.e(), int(1)),
                vec![
                    if_(
                        ne(call("zb_box_get_bool", vec![x.e()], i32()), int32(0)),
                        vec![piece(text("true"))],
                        vec![piece(text("false"))],
                    ),
                    ret_void(),
                ],
            ),
            when(
                or(eq(cat.e(), int(2)), eq(cat.e(), int(3))),
                vec![
                    piece(call(
                        "zb_str_of_int",
                        vec![call("zb_box_get_i64", vec![x.e()], i64())],
                        string(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                eq(cat.e(), int(4)),
                vec![
                    piece(call(
                        "zb_json_float",
                        vec![call("zb_box_get_f64", vec![x.e()], f64())],
                        string(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                eq(cat.e(), int(5)),
                vec![
                    piece(call(
                        "zb_json_escape",
                        vec![call("zb_box_get_str", vec![x.e()], string())],
                        string(),
                    )),
                    ret_void(),
                ],
            ),
            when(
                ne(cat.e(), int(255)),
                vec![fatal(
                    "TypeError",
                    add(
                        add(
                            text("Object of type "),
                            call("zb_any_type", vec![x.e()], string()),
                        ),
                        text(" is not JSON serializable"),
                    ),
                )],
            ),
            // A dict: its key, value pairs after the index slot.
            when(
                eq(kind(x.e()), int(DICT_TAG >> 8)),
                vec![
                    items.decl(raw(x.e())),
                    piece(text("{")),
                    n.decl(mcall(items.e(), "len", vec![], i64())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            when(gt(i.e(), int(1)), vec![piece(text(", "))]),
                            when(
                                ne(category(idx(items.e(), i.e(), any())), int(5)),
                                vec![fatal(
                                    "TypeError",
                                    text("keys must be str, int, float, bool or None"),
                                )],
                            ),
                            into(idx(items.e(), i.e(), any())),
                            piece(text(": ")),
                            into(idx(items.e(), add(i.e(), int(1)), any())),
                            i.add_assign(int(2)),
                        ],
                    ),
                    piece(text("}")),
                    ret_void(),
                ],
            ),
            when(
                eq(kind(x.e()), int(TUPLE_TAG >> 8)),
                vec![block_of(sequence(call(
                    "zb_unbox_tuple_raw",
                    vec![x.e()],
                    anys.clone(),
                )))],
            ),
            // An instance or anything that is not a list.
            when(
                call("zb_any_is_instance", vec![x.e()], boolean()),
                vec![fatal(
                    "TypeError",
                    add(
                        add(
                            text("Object of type "),
                            call("zb_any_type", vec![x.e()], string()),
                        ),
                        text(" is not JSON serializable"),
                    ),
                )],
            ),
            // Any list: its items as dynamic values.
            block_of(sequence(call("zb_any_iter", vec![x.e()], anys.clone()))),
        ],
    ));
    let out = local("pieces", strs.clone());
    d.push(define(
        "zb_json_dumps",
        &[&x],
        string(),
        vec![
            out.decl(list(Vec::new(), strs.clone())),
            expr(call("zb_json_into", vec![out.e(), x.e()], unit())),
            ret(call("zb_str_join", vec![text(""), out.e()], string())),
        ],
    ));
    // Floats print as Python's repr, with the JSON spellings of the
    // values that have no digits.
    let f = local("f", f64());
    d.push(define(
        "zb_json_float",
        &[&f],
        string(),
        vec![
            when(ne(f.e(), f.e()), vec![ret(text("NaN"))]),
            when(eq(f.e(), float(f64::INFINITY)), vec![ret(text("Infinity"))]),
            when(
                eq(f.e(), float(f64::NEG_INFINITY)),
                vec![ret(text("-Infinity"))],
            ),
            ret(call("zb_float_repr", vec![f.e()], string())),
        ],
    ));
    d
}
