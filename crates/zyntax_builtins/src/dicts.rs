//! Dictionaries and sets over lists of dynamic values.
//!
//! A dict is a list holding key, value, key, value in insertion order;
//! a set is a list of distinct values. Lookup compares with the dynamic
//! equality, so both are linear in their size, and both keep the order
//! things were added in, which is what printing and iteration show.

use crate::build::*;
use crate::{list_of, DICT_TAG, SET_TAG};
use zyntax_typed_ast::TypeId;

fn any_eq(a: Expr, b: Expr) -> Expr {
    call("zb_any_eq", vec![a, b], boolean())
}
fn any_repr(x: Expr) -> Expr {
    call("zb_any_repr", vec![x], string())
}
/// A key as an error reports it; the frontend's KeyError quotes it.
fn any_str(x: Expr) -> Expr {
    call("zb_any_str", vec![x], string())
}
fn len(xs: Expr) -> Expr {
    mcall(xs, "len", vec![], i64())
}
fn push(xs: Expr, v: Expr) -> Stmt {
    expr(mcall(xs, "push", vec![v], unit()))
}
fn at(xs: Expr, i: Expr) -> Expr {
    idx(xs, i, any())
}

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let mut d = dict(list_type);
    d.extend(set(list_type));
    d
}

fn dict(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let d = local("d", anys.clone());
    let k = owned("k", any());
    let v = owned("v", any());
    // Returned when the key is absent, so the caller cannot release it.
    let default = owned("default", any());
    let i = local("i", i64());
    let n = local("n", i64());
    let out = local("out", anys.clone());
    let text_out = local("text", string());
    let x = local("x", any());
    let tag = local("tag", i64());
    let mut out_decls = Vec::new();

    // The position of `k`'s pair, or -1.
    out_decls.push(define(
        "zb_dict_find",
        &[&d, &k],
        i64(),
        vec![
            n.decl(len(d.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(any_eq(at(d.e(), i.e()), k.e()), vec![ret(i.e())]),
                    i.add_assign(int(2)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    let find = |k: Expr| call("zb_dict_find", vec![d.e(), k], i64());
    out_decls.push(define(
        "zb_dict_len",
        &[&d],
        i64(),
        vec![ret(div(len(d.e()), int(2)))],
    ));
    out_decls.push(define(
        "zb_dict_contains",
        &[&d, &k],
        boolean(),
        vec![ret(ge(find(k.e()), int(0)))],
    ));
    out_decls.push(define(
        "zb_dict_get",
        &[&d, &k],
        any(),
        vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out_decls.push(define(
        "zb_dict_get_default",
        &[&d, &k, &default],
        any(),
        vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![ret(default.e())]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out_decls.push(define(
        "zb_dict_set",
        &[&d, &k, &v],
        unit(),
        vec![
            i.decl(find(k.e())),
            if_(
                lt(i.e(), int(0)),
                vec![push(d.e(), k.e()), push(d.e(), v.e())],
                vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
            ),
            ret_void(),
        ],
    ));
    // `setdefault`: the value present, or `v` stored and returned.
    out_decls.push(define(
        "zb_dict_setdefault",
        &[&d, &k, &v],
        any(),
        vec![
            i.decl(find(k.e())),
            when(ge(i.e(), int(0)), vec![ret(at(d.e(), add(i.e(), int(1))))]),
            push(d.e(), k.e()),
            push(d.e(), v.e()),
            ret(v.e()),
        ],
    ));
    let remove_pair = |i: &Local| {
        vec![
            expr(mcall(d.e(), "remove_at", vec![add(i.e(), int(1))], any())),
            expr(mcall(d.e(), "remove_at", vec![i.e()], any())),
        ]
    };
    out_decls.push(define("zb_dict_del", &[&d, &k], unit(), {
        let mut s = vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
        ];
        s.extend(remove_pair(&i));
        s.push(ret_void());
        s
    }));
    out_decls.push(define("zb_dict_pop", &[&d, &k], any(), {
        let mut s = vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
            v.decl(at(d.e(), add(i.e(), int(1)))),
        ];
        s.extend(remove_pair(&i));
        s.push(ret(v.e()));
        s
    }));
    out_decls.push(define("zb_dict_pop_default", &[&d, &k, &default], any(), {
        let mut s = vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![ret(default.e())]),
            v.decl(at(d.e(), add(i.e(), int(1)))),
        ];
        s.extend(remove_pair(&i));
        s.push(ret(v.e()));
        s
    }));
    // Every key, every value, every (key, value) tuple, in order.
    for (name, offset) in [("zb_dict_keys", 0), ("zb_dict_values", 1)] {
        out_decls.push(define(name, &[&d], anys.clone(), {
            let mut s = vec![
                out.decl(list(Vec::new(), anys.clone())),
                n.decl(len(d.e())),
                i.decl(int(0)),
            ];
            s.push(while_(
                lt(i.e(), n.e()),
                vec![
                    push(out.e(), at(d.e(), add(i.e(), int(offset)))),
                    i.add_assign(int(2)),
                ],
            ));
            s.push(ret(out.e()));
            s
        }));
    }
    let pair = local("pair", anys.clone());
    out_decls.push(define("zb_dict_items", &[&d], anys.clone(), {
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(d.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    pair.decl(list(
                        vec![at(d.e(), i.e()), at(d.e(), add(i.e(), int(1)))],
                        anys.clone(),
                    )),
                    push(out.e(), call("zb_box_tuple", vec![pair.e()], any())),
                    i.add_assign(int(2)),
                ],
            ),
            ret(out.e()),
        ]
    }));
    let other = local("other", anys.clone());
    out_decls.push(define("zb_dict_update", &[&d, &other], unit(), {
        vec![
            n.decl(len(other.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    expr(call(
                        "zb_dict_set",
                        vec![
                            d.e(),
                            at(other.e(), i.e()),
                            at(other.e(), add(i.e(), int(1))),
                        ],
                        unit(),
                    )),
                    i.add_assign(int(2)),
                ],
            ),
            ret_void(),
        ]
    }));
    // A dict from a literal's keys and values, later pairs winning.
    let pairs = local("pairs", anys.clone());
    out_decls.push(define("zb_dict_from_pairs", &[&pairs], anys.clone(), {
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            expr(call("zb_dict_update", vec![out.e(), pairs.e()], unit())),
            ret(out.e()),
        ]
    }));
    out_decls.push(define(
        "zb_dict_copy",
        &[&d],
        anys.clone(),
        vec![ret(call("zb_list_copy_any", vec![d.e()], anys.clone()))],
    ));
    // Equal when every pair of one is in the other and sizes match.
    let j = local("j", i64());
    out_decls.push(define(
        "zb_dict_eq",
        &[&d, &other],
        boolean(),
        vec![
            n.decl(len(d.e())),
            when(ne(n.e(), len(other.e())), vec![ret(bool(false))]),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    j.decl(call(
                        "zb_dict_find",
                        vec![other.e(), at(d.e(), i.e())],
                        i64(),
                    )),
                    when(lt(j.e(), int(0)), vec![ret(bool(false))]),
                    when(
                        not(any_eq(
                            at(d.e(), add(i.e(), int(1))),
                            at(other.e(), add(j.e(), int(1))),
                        )),
                        vec![ret(bool(false))],
                    ),
                    i.add_assign(int(2)),
                ],
            ),
            ret(bool(true)),
        ],
    ));
    out_decls.push(define("zb_dict_repr", &[&d], string(), {
        vec![
            text_out.decl(text("{")),
            n.decl(len(d.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(
                        gt(i.e(), int(0)),
                        vec![text_out.set(add(text_out.e(), text(", ")))],
                    ),
                    text_out.set(add(
                        add(add(text_out.e(), any_repr(at(d.e(), i.e()))), text(": ")),
                        any_repr(at(d.e(), add(i.e(), int(1)))),
                    )),
                    i.add_assign(int(2)),
                ],
            ),
            ret(add(text_out.e(), text("}"))),
        ]
    }));
    out_decls.push(extern_fn(
        "zb_box_dict_raw",
        &[("d", anys.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    out_decls.push(define(
        "zb_dict_box",
        &[&d],
        any(),
        vec![ret(call(
            "zb_box_dict_raw",
            vec![d.e(), int32(DICT_TAG as i32)],
            any(),
        ))],
    ));
    out_decls.push(define(
        "zb_dict_unbox",
        &[&x],
        anys.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64())),
            when(
                ne(tag.e(), int(DICT_TAG)),
                vec![fatal(
                    "TypeError",
                    add(
                        text("expected a dict, got "),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call("zb_unbox_list_raw_any", vec![x.e()], anys.clone())),
        ],
    ));
    out_decls
}

fn set(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let s = local("s", anys.clone());
    let other = local("other", anys.clone());
    let v = owned("v", any());
    let i = local("i", i64());
    let n = local("n", i64());
    let out = local("out", anys.clone());
    let text_out = local("text", string());
    let x = local("x", any());
    let tag = local("tag", i64());
    let contains = |s: Expr, v: Expr| call("zb_list_contains_any", vec![s, v], boolean());
    let mut d = Vec::new();

    d.push(define(
        "zb_set_add",
        &[&s, &v],
        unit(),
        vec![
            when(not(contains(s.e(), v.e())), vec![push(s.e(), v.e())]),
            ret_void(),
        ],
    ));
    d.push(define(
        "zb_set_discard",
        &[&s, &v],
        unit(),
        vec![
            i.decl(call("zb_list_index_or_neg_any", vec![s.e(), v.e()], i64())),
            when(
                ge(i.e(), int(0)),
                vec![expr(mcall(s.e(), "remove_at", vec![i.e()], any()))],
            ),
            ret_void(),
        ],
    ));
    d.push(define(
        "zb_set_remove",
        &[&s, &v],
        unit(),
        vec![
            i.decl(call("zb_list_index_or_neg_any", vec![s.e(), v.e()], i64())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(v.e()))]),
            expr(mcall(s.e(), "remove_at", vec![i.e()], any())),
            ret_void(),
        ],
    ));
    // A set from any list, keeping first occurrences.
    let xs = local("xs", anys.clone());
    d.push(define("zb_set_from", &[&xs], anys.clone(), {
        let mut st = vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(xs.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_add",
                vec![out.e(), at(xs.e(), i.e())],
                unit(),
            ))],
        ));
        st.push(ret(out.e()));
        st
    }));
    // Intersection, union, difference, symmetric difference.
    d.push(define("zb_set_and", &[&s, &other], anys.clone(), {
        let mut st = vec![out.decl(list(Vec::new(), anys.clone())), n.decl(len(s.e()))];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(
                contains(other.e(), at(s.e(), i.e())),
                vec![push(out.e(), at(s.e(), i.e()))],
            )],
        ));
        st.push(ret(out.e()));
        st
    }));
    d.push(define("zb_set_or", &[&s, &other], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_list_copy_any", vec![s.e()], anys.clone())),
            n.decl(len(other.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_add",
                vec![out.e(), at(other.e(), i.e())],
                unit(),
            ))],
        ));
        st.push(ret(out.e()));
        st
    }));
    d.push(define("zb_set_sub", &[&s, &other], anys.clone(), {
        let mut st = vec![out.decl(list(Vec::new(), anys.clone())), n.decl(len(s.e()))];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(
                not(contains(other.e(), at(s.e(), i.e()))),
                vec![push(out.e(), at(s.e(), i.e()))],
            )],
        ));
        st.push(ret(out.e()));
        st
    }));
    d.push(define(
        "zb_set_xor",
        &[&s, &other],
        anys.clone(),
        vec![ret(call(
            "zb_set_or",
            vec![
                call("zb_set_sub", vec![s.e(), other.e()], anys.clone()),
                call("zb_set_sub", vec![other.e(), s.e()], anys.clone()),
            ],
            anys.clone(),
        ))],
    ));
    d.push(define(
        "zb_set_issubset",
        &[&s, &other],
        boolean(),
        vec![ret(eq(
            len(call("zb_set_sub", vec![s.e(), other.e()], anys.clone())),
            int(0),
        ))],
    ));
    d.push(define(
        "zb_set_eq",
        &[&s, &other],
        boolean(),
        vec![ret(and(
            eq(len(s.e()), len(other.e())),
            call("zb_set_issubset", vec![s.e(), other.e()], boolean()),
        ))],
    ));
    d.push(define(
        "zb_set_repr",
        &[&s],
        string(),
        vec![
            when(eq(len(s.e()), int(0)), vec![ret(text("set()"))]),
            text_out.decl(call(
                "zb_list_items_any",
                vec![s.e(), text("{"), text("}")],
                string(),
            )),
            ret(text_out.e()),
        ],
    ));
    d.push(extern_fn(
        "zb_box_set_raw",
        &[("s", anys.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    d.push(define(
        "zb_set_box",
        &[&s],
        any(),
        vec![ret(call(
            "zb_box_set_raw",
            vec![s.e(), int32(SET_TAG as i32)],
            any(),
        ))],
    ));
    d.push(define(
        "zb_set_unbox",
        &[&x],
        anys.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64())),
            when(
                ne(tag.e(), int(SET_TAG)),
                vec![fatal(
                    "TypeError",
                    add(
                        text("expected a set, got "),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call("zb_unbox_list_raw_any", vec![x.e()], anys.clone())),
        ],
    ));
    d
}
