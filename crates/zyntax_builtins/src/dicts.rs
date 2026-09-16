//! Dictionaries and sets over lists of dynamic values.
//!
//! A dict is a list whose first element is its hash index and the rest
//! key, value, key, value in insertion order, so iteration and printing
//! show the order things were added in and a lookup is a hash and a
//! probe. A set is a list of distinct values, searched linearly.

use crate::build::*;
use crate::{list_of, DICT_TAG, SET_TAG, TUPLE_TAG};
use zyntax_typed_ast::TypeId;

fn any_eq(a: Expr, b: Expr) -> Expr {
    call("zb_any_eq", vec![a, b], boolean())
}
/// Whether a stored key matches the one looked up: the same box, as a
/// key read back from the dict or a shared constant is, before the
/// values are compared.
fn key_matches(stored: Expr, k: Expr) -> Expr {
    or(eq(stored.clone(), k.clone()), any_eq(stored, k))
}
/// The statements that return `found` when `stored` matches `k`, whose
/// category is `kcat`: the same box, or two strings compared as
/// strings, or anything else compared as dynamic values.
fn when_key_matches(stored: &Local, k: &Local, kcat: &Local, found: Expr) -> Vec<Stmt> {
    let category = |x: Expr| call("zb_any_category", vec![x], i64());
    let text = |x: Expr| call("zb_box_get_str", vec![x], string());
    vec![
        when(eq(stored.e(), k.e()), vec![ret(found.clone())]),
        if_(
            and(
                eq(kcat.e(), int(crate::dynamic::STR)),
                eq(category(stored.e()), int(crate::dynamic::STR)),
            ),
            vec![when(
                call("zb_str_eq", vec![text(stored.e()), text(k.e())], boolean()),
                vec![ret(found.clone())],
            )],
            vec![when(any_eq(stored.e(), k.e()), vec![ret(found)])],
        ),
    ]
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
    let ints = list_of(list_type, i64());
    // Every function edits the dict in place; only boxing keeps it.
    let d = borrowed("d", anys.clone());
    let k = kept("k", any());
    // The lookups read the key and keep nothing, so a box made for one
    // is the caller's to release straight after the call.
    let key = borrowed("k", any());
    let v = kept("v", any());
    // Returned when the key is absent, so the caller cannot release it.
    let default = kept("default", any());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());
    let out = local("out", anys.clone());
    let x = local("x", any());
    let tag = local("tag", i64());
    let index = borrowed("index", ints.clone());
    let mask = local("mask", i64());
    let slot = local("s", i64());
    let entry = local("e", i64());
    let cap = local("cap", i64());
    let h = local("h", i64());
    let kcat = local("kcat", i64());
    let stored = local("stored", any());
    let mut out_decls = Vec::new();

    // The index is a power-of-two table of entry numbers, -1 where
    // empty, probed linearly from the key's hash and kept under half
    // full. Entry `e` is the pair at positions 1 + 2e and 2 + 2e. A
    // dict of a few pairs has no table (its slot holds None) and is
    // scanned, which costs less than hashing for so few.
    let index_of = |d: Expr| call("zb_unbox_list_raw_i64", vec![at(d, int(0))], ints.clone());
    let box_index = |index: Expr| call("zb_list_box_i64", vec![index], any());
    let slot_at = |index: Expr, s: Expr| idx(index, s, i64());
    let hash = |k: Expr| call("zb_dict_hash", vec![k], i64());
    let count = |d: Expr| div(sub(len(d), int(1)), int(2));
    let key_at = |d: Expr, e: Expr| at(d, add(mul(e, int(2)), int(1)));
    let next_slot = |s: Expr, mask: Expr| bitand(add(s, int(1)), mask);
    let unindexed = |d: Expr| eq(at(d, int(0)), null(any()));
    /// Pairs a dict holds before it takes a table.
    const SMALL: i64 = 8;
    /// The first table's size, for a dict just past `SMALL`.
    const FIRST_TABLE: i64 = 32;

    // The hash a key lands by: the value's hash with its bits spread,
    // since the table takes the low ones.
    out_decls.push(define(
        "zb_dict_hash",
        &[&k],
        i64(),
        vec![
            h.decl(call("zb_any_hash", vec![k.e()], i64())),
            h.set(bitxor(h.e(), shr(h.e(), int(32)))),
            h.set(mul(h.e(), int(-7_046_029_254_386_353_131))),
            h.set(bitxor(h.e(), shr(h.e(), int(29)))),
            ret(h.e()),
        ],
    ));
    // An empty table of `cap` slots.
    out_decls.push(define("zb_dict_index_new", &[&cap], ints.clone(), {
        let table = local("table", ints.clone());
        let mut st = vec![
            table.decl(list(Vec::new(), ints.clone())),
            expr(mcall(table.e(), "reserve", vec![cap.e()], unit())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            cap.e(),
            vec![expr(mcall(table.e(), "push", vec![int(-1)], unit()))],
        ));
        st.push(ret(table.e()));
        st
    }));
    // Record entry `e`, whose key hashes to `h`, in `index`, which has
    // room for it.
    out_decls.push(define(
        "zb_dict_place_hashed",
        &[&index, &entry, &h],
        unit(),
        vec![
            mask.decl(sub(len(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            while_(
                ge(slot_at(index.e(), slot.e()), int(0)),
                vec![slot.set(next_slot(slot.e(), mask.e()))],
            ),
            set_idx(index.e(), slot.e(), entry.e()),
            ret_void(),
        ],
    ));
    out_decls.push(define(
        "zb_dict_place",
        &[&index, &d, &entry],
        unit(),
        vec![
            expr(call(
                "zb_dict_place_hashed",
                vec![index.e(), entry.e(), hash(key_at(d.e(), entry.e()))],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // A fresh table of `cap` slots over every entry of `d`.
    out_decls.push(define("zb_dict_reindex", &[&d, &cap], unit(), {
        let mut st = vec![
            index.decl(call("zb_dict_index_new", vec![cap.e()], ints.clone())),
            n.decl(count(d.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_dict_place",
                vec![index.e(), d.e(), i.e()],
                unit(),
            ))],
        ));
        st.push(set_idx(d.e(), int(0), box_index(index.e())));
        st.push(ret_void());
        st
    }));
    out_decls.push(define(
        "zb_dict_new",
        &[],
        anys.clone(),
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            push(out.e(), null(any())),
            ret(out.e()),
        ],
    ));
    // The position of `k`'s key in a dict with a table, given the
    // key's hash, or -1 (bounded by the table's size, which a table
    // under half full never reaches); and the position of `k`'s key in
    // any dict, or -1. Defined twice: comparing two dicts looks keys
    // up and compares values, and comparing values can compare dicts,
    // so the lookups equality uses are its own, and the ones
    // everything else uses stay outside that cycle and inline into
    // their callers.
    for suffix in ["", "_eq"] {
        let find_hashed = format!("zb_dict_find_hashed{suffix}");
        out_decls.push(define(
            &find_hashed,
            &[&d, &key, &h],
            i64(),
            vec![
                index.decl(index_of(d.e())),
                mask.decl(sub(len(index.e()), int(1))),
                slot.decl(bitand(h.e(), mask.e())),
                kcat.decl(call("zb_any_category", vec![k.e()], i64())),
                i.decl(int(0)),
                while_(le(i.e(), mask.e()), {
                    let mut body = vec![
                        entry.decl(slot_at(index.e(), slot.e())),
                        when(lt(entry.e(), int(0)), vec![ret(int(-1))]),
                        stored.decl(key_at(d.e(), entry.e())),
                    ];
                    body.extend(when_key_matches(
                        &stored,
                        &k,
                        &kcat,
                        add(mul(entry.e(), int(2)), int(1)),
                    ));
                    body.push(slot.set(next_slot(slot.e(), mask.e())));
                    body.push(i.add_assign(int(1)));
                    body
                }),
                ret(int(-1)),
            ],
        ));
        out_decls.push(define(
            &format!("zb_dict_find{suffix}"),
            &[&d, &key],
            i64(),
            vec![
                when(
                    unindexed(d.e()),
                    vec![
                        n.decl(len(d.e())),
                        i.decl(int(1)),
                        while_(
                            lt(i.e(), n.e()),
                            vec![
                                when(key_matches(at(d.e(), i.e()), k.e()), vec![ret(i.e())]),
                                i.add_assign(int(2)),
                            ],
                        ),
                        ret(int(-1)),
                    ],
                ),
                ret(call(&find_hashed, vec![d.e(), k.e(), hash(k.e())], i64())),
            ],
        ));
    }
    let find = |k: Expr| call("zb_dict_find", vec![d.e(), k], i64());
    // Add a pair whose key is absent and hashes to `h`, growing the
    // table first when the pair would bring it to half full.
    out_decls.push(define(
        "zb_dict_insert_hashed",
        &[&d, &k, &v, &h],
        unit(),
        vec![
            entry.decl(count(d.e())),
            cap.decl(len(index_of(d.e()))),
            when(
                gt(mul(add(entry.e(), int(1)), int(2)), cap.e()),
                vec![expr(call(
                    "zb_dict_reindex",
                    vec![d.e(), mul(cap.e(), int(2))],
                    unit(),
                ))],
            ),
            push(d.e(), k.e()),
            push(d.e(), v.e()),
            expr(call(
                "zb_dict_place_hashed",
                vec![index_of(d.e()), entry.e(), h.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // Add a pair whose key is absent; a dict that has grown past a few
    // pairs takes its first table.
    out_decls.push(define(
        "zb_dict_insert",
        &[&d, &k, &v],
        unit(),
        vec![
            when(
                unindexed(d.e()),
                vec![
                    push(d.e(), k.e()),
                    push(d.e(), v.e()),
                    when(
                        gt(count(d.e()), int(SMALL)),
                        vec![expr(call(
                            "zb_dict_reindex",
                            vec![d.e(), int(FIRST_TABLE)],
                            unit(),
                        ))],
                    ),
                    ret_void(),
                ],
            ),
            expr(call(
                "zb_dict_insert_hashed",
                vec![d.e(), k.e(), v.e(), hash(k.e())],
                unit(),
            )),
            ret_void(),
        ],
    ));
    let insert = |k: Expr, v: Expr| expr(call("zb_dict_insert", vec![d.e(), k, v], unit()));
    out_decls.push(define("zb_dict_len", &[&d], i64(), vec![ret(count(d.e()))]));

    // Lookups by a string that is not boxed, for a key the frontend
    // knows to be one: hashed and compared as a string, so a key
    // already present costs no box, and one stored is boxed then.
    let s = borrowed("text", string());
    let stored_text = |x: Expr| call("zb_box_get_str", vec![x], string());
    // The text is read only once the box is known to hold one.
    let when_stored_is = |stored: &Local, found: Expr| {
        when(
            eq(
                call("zb_any_category", vec![stored.e()], i64()),
                int(crate::dynamic::STR),
            ),
            vec![when(
                call("zb_str_eq", vec![stored_text(stored.e()), s.e()], boolean()),
                vec![ret(found)],
            )],
        )
    };
    out_decls.push(define(
        "zb_dict_str_hash",
        &[&s],
        i64(),
        vec![
            h.decl(call("zb_str_hash", vec![s.e()], i64())),
            when(eq(h.e(), int(0)), vec![h.set(int(1))]),
            h.set(bitxor(h.e(), shr(h.e(), int(32)))),
            h.set(mul(h.e(), int(-7_046_029_254_386_353_131))),
            h.set(bitxor(h.e(), shr(h.e(), int(29)))),
            ret(h.e()),
        ],
    ));
    out_decls.push(define(
        "zb_dict_find_hashed_str",
        &[&d, &s, &h],
        i64(),
        vec![
            index.decl(index_of(d.e())),
            mask.decl(sub(len(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    entry.decl(slot_at(index.e(), slot.e())),
                    when(lt(entry.e(), int(0)), vec![ret(int(-1))]),
                    stored.decl(key_at(d.e(), entry.e())),
                    when_stored_is(&stored, add(mul(entry.e(), int(2)), int(1))),
                    slot.set(next_slot(slot.e(), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    out_decls.push(define(
        "zb_dict_find_str",
        &[&d, &s],
        i64(),
        vec![
            when(
                unindexed(d.e()),
                vec![
                    n.decl(len(d.e())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            stored.decl(at(d.e(), i.e())),
                            when_stored_is(&stored, i.e()),
                            i.add_assign(int(2)),
                        ],
                    ),
                    ret(int(-1)),
                ],
            ),
            ret(call(
                "zb_dict_find_hashed_str",
                vec![d.e(), s.e(), call("zb_dict_str_hash", vec![s.e()], i64())],
                i64(),
            )),
        ],
    ));
    let find_str = || call("zb_dict_find_str", vec![d.e(), s.e()], i64());
    let boxed_key = || call("zb_str_to_dynamic", vec![s.e()], any());
    out_decls.push(define(
        "zb_dict_contains_str",
        &[&d, &s],
        boolean(),
        vec![ret(ge(find_str(), int(0)))],
    ));
    out_decls.push(define(
        "zb_dict_get_str",
        &[&d, &s],
        any(),
        vec![
            i.decl(find_str()),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", s.e())]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out_decls.push(define(
        "zb_dict_get_default_str",
        &[&d, &s, &default],
        any(),
        vec![
            i.decl(find_str()),
            when(lt(i.e(), int(0)), vec![ret(default.e())]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out_decls.push(define(
        "zb_dict_set_str",
        &[&d, &s, &v],
        unit(),
        vec![
            when(
                unindexed(d.e()),
                vec![
                    i.decl(find_str()),
                    if_(
                        lt(i.e(), int(0)),
                        vec![insert(boxed_key(), v.e())],
                        vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(call("zb_dict_str_hash", vec![s.e()], i64())),
            i.decl(call(
                "zb_dict_find_hashed_str",
                vec![d.e(), s.e(), h.e()],
                i64(),
            )),
            if_(
                lt(i.e(), int(0)),
                vec![expr(call(
                    "zb_dict_insert_hashed",
                    vec![d.e(), boxed_key(), v.e(), h.e()],
                    unit(),
                ))],
                vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
            ),
            ret_void(),
        ],
    ));
    out_decls.push(define(
        "zb_dict_contains",
        &[&d, &key],
        boolean(),
        vec![ret(ge(find(k.e()), int(0)))],
    ));
    out_decls.push(define(
        "zb_dict_get",
        &[&d, &key],
        any(),
        vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out_decls.push(define(
        "zb_dict_get_default",
        &[&d, &key, &default],
        any(),
        vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![ret(default.e())]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    // Store `v` under `k`. The key is hashed once on the table path.
    out_decls.push(define(
        "zb_dict_set",
        &[&d, &k, &v],
        unit(),
        vec![
            when(
                unindexed(d.e()),
                vec![
                    i.decl(find(k.e())),
                    if_(
                        lt(i.e(), int(0)),
                        vec![insert(k.e(), v.e())],
                        vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(hash(k.e())),
            i.decl(call(
                "zb_dict_find_hashed",
                vec![d.e(), k.e(), h.e()],
                i64(),
            )),
            if_(
                lt(i.e(), int(0)),
                vec![expr(call(
                    "zb_dict_insert_hashed",
                    vec![d.e(), k.e(), v.e(), h.e()],
                    unit(),
                ))],
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
            insert(k.e(), v.e()),
            ret(v.e()),
        ],
    ));
    // Removal shifts the later pairs down, so the table is rebuilt.
    let remove_pair = |i: &Local| {
        vec![
            expr(mcall(d.e(), "remove_at", vec![add(i.e(), int(1))], any())),
            expr(mcall(d.e(), "remove_at", vec![i.e()], any())),
            when(
                not(unindexed(d.e())),
                vec![expr(call(
                    "zb_dict_reindex",
                    vec![d.e(), len(index_of(d.e()))],
                    unit(),
                ))],
            ),
        ]
    };
    out_decls.push(define("zb_dict_del", &[&d, &key], unit(), {
        let mut s = vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
        ];
        s.extend(remove_pair(&i));
        s.push(ret_void());
        s
    }));
    out_decls.push(define("zb_dict_pop", &[&d, &key], any(), {
        let mut s = vec![
            i.decl(find(k.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(k.e()))]),
            v.decl(at(d.e(), add(i.e(), int(1)))),
        ];
        s.extend(remove_pair(&i));
        s.push(ret(v.e()));
        s
    }));
    out_decls.push(define(
        "zb_dict_pop_default",
        &[&d, &key, &default],
        any(),
        {
            let mut s = vec![
                i.decl(find(k.e())),
                when(lt(i.e(), int(0)), vec![ret(default.e())]),
                v.decl(at(d.e(), add(i.e(), int(1)))),
            ];
            s.extend(remove_pair(&i));
            s.push(ret(v.e()));
            s
        },
    ));
    // Every key, every value, every (key, value) tuple, in order.
    for (name, offset) in [("zb_dict_keys", 0), ("zb_dict_values", 1)] {
        out_decls.push(define(name, &[&d], anys.clone(), {
            let mut s = vec![
                out.decl(list(Vec::new(), anys.clone())),
                n.decl(len(d.e())),
                i.decl(int(1)),
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
            i.decl(int(1)),
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
    let other = borrowed("other", anys.clone());
    out_decls.push(define("zb_dict_update", &[&d, &other], unit(), {
        vec![
            n.decl(len(other.e())),
            i.decl(int(1)),
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
    let pairs = borrowed("pairs", anys.clone());
    out_decls.push(define("zb_dict_add_pairs", &[&d, &pairs], unit(), {
        vec![
            n.decl(len(pairs.e())),
            i.decl(int(0)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    expr(call(
                        "zb_dict_set",
                        vec![
                            d.e(),
                            at(pairs.e(), i.e()),
                            at(pairs.e(), add(i.e(), int(1))),
                        ],
                        unit(),
                    )),
                    i.add_assign(int(2)),
                ],
            ),
            ret_void(),
        ]
    }));
    out_decls.push(define(
        "zb_dict_from_pairs",
        &[&pairs],
        anys.clone(),
        vec![
            out.decl(call("zb_dict_new", vec![], anys.clone())),
            expr(call("zb_dict_add_pairs", vec![out.e(), pairs.e()], unit())),
            ret(out.e()),
        ],
    ));
    // A dict over storage a literal laid out itself: the index slot,
    // then pairs whose keys are known to be distinct. The storage is
    // the dict; only a large one takes a table.
    out_decls.push(define(
        "zb_dict_from_distinct",
        &[&d],
        anys.clone(),
        vec![
            n.decl(count(d.e())),
            when(
                gt(n.e(), int(SMALL)),
                vec![
                    cap.decl(int(FIRST_TABLE)),
                    while_(
                        lt(cap.e(), mul(n.e(), int(2))),
                        vec![cap.set(mul(cap.e(), int(2)))],
                    ),
                    expr(call("zb_dict_reindex", vec![d.e(), cap.e()], unit())),
                ],
            ),
            ret(d.e()),
        ],
    ));
    out_decls.push(define(
        "zb_dict_copy",
        &[&d],
        anys.clone(),
        vec![
            out.decl(call("zb_dict_new", vec![], anys.clone())),
            expr(call("zb_dict_update", vec![out.e(), d.e()], unit())),
            ret(out.e()),
        ],
    ));
    // Equal when every pair of one is in the other and sizes match.
    out_decls.push(define(
        "zb_dict_eq",
        &[&d, &other],
        boolean(),
        vec![
            n.decl(len(d.e())),
            when(ne(n.e(), len(other.e())), vec![ret(bool(false))]),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    j.decl(call(
                        "zb_dict_find_eq",
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
    // Pieces joined once, so the text costs what its length is.
    let pieces = local("pieces", list_of(list_type, string()));
    let piece = |p: Expr| expr(mcall(pieces.e(), "push", vec![p], unit()));
    out_decls.push(define("zb_dict_repr", &[&d], string(), {
        vec![
            pieces.decl(list(Vec::new(), list_of(list_type, string()))),
            piece(text("{")),
            n.decl(len(d.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(gt(i.e(), int(1)), vec![piece(text(", "))]),
                    piece(any_repr(at(d.e(), i.e()))),
                    piece(text(": ")),
                    piece(any_repr(at(d.e(), add(i.e(), int(1))))),
                    i.add_assign(int(2)),
                ],
            ),
            piece(text("}")),
            ret(call("zb_str_join", vec![text(""), pieces.e()], string())),
        ]
    }));
    out_decls.push(extern_fn(
        "zb_box_dict_raw",
        &[("d", anys.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    let boxed = local("d", anys.clone());
    out_decls.push(define(
        "zb_dict_box",
        &[&boxed],
        any(),
        vec![ret(call(
            "zb_box_dict_raw",
            vec![boxed.e(), int32(DICT_TAG as i32)],
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
    let v = kept("v", any());
    // Discarding reads the value and keeps nothing.
    let value = borrowed("v", any());
    let i = local("i", i64());
    let n = local("n", i64());
    let out = local("out", anys.clone());
    let text_out = local("text", string());
    let x = local("x", any());
    let tag = local("tag", i64());
    let mask = local("mask", i64());
    let left_mask = local("left_mask", i64());
    let right_mask = local("right_mask", i64());
    let member = local("member", any());
    let bit = local("bit", i64());
    let wanted = local("wanted", i64());
    let contains = |s: Expr, v: Expr| call("zb_list_contains_any", vec![s, v], boolean());
    let mut d = Vec::new();

    // Integer board positions fit in a word. -1 means the set also
    // holds a value that needs the general equality path.
    d.push(define("zb_set_mask63", &[&s], i64(), {
        let mut st = vec![mask.decl(int(0)), n.decl(len(s.e()))];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                member.decl(at(s.e(), i.e())),
                when(
                    ne(
                        cast(call("zb_box_tag", vec![member.e()], i32()), i64()),
                        int(crate::dynamic::I64_TAG),
                    ),
                    vec![ret(int(-1))],
                ),
                bit.decl(call("zb_box_get_i64", vec![member.e()], i64())),
                when(
                    or(lt(bit.e(), int(0)), ge(bit.e(), int(63))),
                    vec![ret(int(-1))],
                ),
                mask.set(bitor(mask.e(), shl(int(1), bit.e()))),
            ],
        ));
        st.push(ret(mask.e()));
        st
    }));
    d.push(define("zb_set_all_kind", &[&s, &wanted], boolean(), {
        let mut st = vec![
            n.decl(len(s.e())),
            when(eq(n.e(), int(0)), vec![ret(bool(false))]),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(
                ne(
                    call("zb_any_kind", vec![at(s.e(), i.e())], i64()),
                    wanted.e(),
                ),
                vec![ret(bool(false))],
            )],
        ));
        st.push(ret(bool(true)));
        st
    }));
    d.push(define(
        "zb_set_disjoint_kinds",
        &[&s, &other],
        boolean(),
        vec![
            when(
                call("zb_set_all_kind", vec![s.e(), int(SET_TAG >> 8)], boolean()),
                vec![ret(call(
                    "zb_set_all_kind",
                    vec![other.e(), int(TUPLE_TAG >> 8)],
                    boolean(),
                ))],
            ),
            when(
                call(
                    "zb_set_all_kind",
                    vec![s.e(), int(TUPLE_TAG >> 8)],
                    boolean(),
                ),
                vec![ret(call(
                    "zb_set_all_kind",
                    vec![other.e(), int(SET_TAG >> 8)],
                    boolean(),
                ))],
            ),
            ret(bool(false)),
        ],
    ));

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
        &[&s, &value],
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
        &[&s, &value],
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
        let mut st = vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut matched = for_range(
                    &i,
                    int(0),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_get_i64", vec![member.e()], i64())),
                        when(
                            ne(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                matched.push(ret(out.e()));
                fast.push(when(ge(right_mask.e(), int(0)), matched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(out.e())],
            ),
        ];
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
        let mut st = vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut unmatched = for_range(
                    &i,
                    int(0),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_get_i64", vec![member.e()], i64())),
                        when(
                            eq(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                unmatched.push(ret(out.e()));
                fast.push(when(ge(right_mask.e(), int(0)), unmatched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                {
                    let mut copy =
                        for_range(&i, int(0), n.e(), vec![push(out.e(), at(s.e(), i.e()))]);
                    copy.push(ret(out.e()));
                    copy
                },
            ),
        ];
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
    d.push(define("zb_set_issubset", &[&s, &other], boolean(), {
        let mut st = vec![
            when(gt(len(s.e()), len(other.e())), vec![ret(bool(false))]),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(
                ge(left_mask.e(), int(0)),
                vec![
                    right_mask.decl(call("zb_set_mask63", vec![other.e()], i64())),
                    when(
                        ge(right_mask.e(), int(0)),
                        vec![ret(eq(
                            bitand(left_mask.e(), right_mask.e()),
                            left_mask.e(),
                        ))],
                    ),
                ],
            ),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(bool(false))],
            ),
            n.decl(len(s.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![when(
                not(contains(other.e(), at(s.e(), i.e()))),
                vec![ret(bool(false))],
            )],
        ));
        st.push(ret(bool(true)));
        st
    }));
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
