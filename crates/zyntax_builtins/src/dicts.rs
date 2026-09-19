//! Dictionaries and sets over lists of dynamic values.
//!
//! A dict is a list whose first element is its hash index and the rest
//! key, value, key, value in insertion order, so iteration and printing
//! show the order things were added in and a lookup is a hash and a
//! probe. A set is laid out the same way over its values.

use crate::build::*;
use crate::{DICT_TAG, SET_TAG, TUPLE_TAG, list_of};
use zyntax_typed_ast::{Type, TypeId};

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
    let hv = local("h", i64());
    out_decls.push(define(
        "zb_hash_mix",
        &[&hv],
        i64(),
        vec![
            h.decl(bitxor(hv.e(), shr(hv.e(), int(32)))),
            h.set(mul(h.e(), int(-7_046_029_254_386_353_131))),
            h.set(bitxor(h.e(), shr(h.e(), int(29)))),
            ret(h.e()),
        ],
    ));
    out_decls.push(define(
        "zb_dict_hash",
        &[&k],
        i64(),
        vec![ret(call(
            "zb_hash_mix",
            vec![call("zb_any_hash", vec![k.e()], i64())],
            i64(),
        ))],
    ));
    // An empty table of `cap` slots: every word -1, the empty mark.
    out_decls.push(define("zb_dict_index_new", &[&cap], ints.clone(), {
        let table = local("table", ints.clone());
        vec![
            table.decl(list(Vec::new(), ints.clone())),
            expr(mcall(
                table.e(),
                "resize_filled",
                vec![cap.e(), int(0xFF)],
                unit(),
            )),
            ret(table.e()),
        ]
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
    out_decls.push(define(
        "zb_dict_as_box",
        &[&x],
        any(),
        vec![
            expr(call("zb_dict_unbox", vec![x.e()], anys.clone())),
            ret(x.e()),
        ],
    ));
    out_decls
}

/// The dict operations keyed by a value of `key_ty` that is not a box:
/// `zb_dict_{find,contains,get,get_default,set}_<suffix>`. `hash_of` is
/// the key's hash as its boxed form hashes, `matches(stored, key)`
/// compares a stored key box with it, `boxed(key)` is the box a stored
/// key becomes, and `text(key)` names it in a KeyError.
pub(crate) fn dict_ops_by(
    list_type: TypeId,
    suffix: &str,
    key_ty: Type,
    hash_of: &dyn Fn(Expr) -> Expr,
    matches: &dyn Fn(Expr, Expr) -> Expr,
    boxed: &dyn Fn(Expr) -> Expr,
    text: &dyn Fn(Expr) -> Expr,
) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let ints = list_of(list_type, i64());
    let d = borrowed("d", anys.clone());
    let key = borrowed("key", key_ty);
    let v = kept("v", any());
    let default = kept("default", any());
    let n = local("n", i64());
    let i = local("i", i64());
    let index = local("index", ints.clone());
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let entry = local("e", i64());
    let h = local("h", i64());
    let name = |op: &str| format!("zb_dict_{op}_{suffix}");
    let mut out = Vec::new();
    // The key's position, or -1, given its mixed hash; the table's
    // slots hold entry numbers, keys sit at 2e + 1.
    out.push(define(
        &name("find_hashed"),
        &[&d, &key, &h],
        i64(),
        vec![
            index.decl(call(
                "zb_unbox_list_raw_i64",
                vec![at(d.e(), int(0))],
                ints.clone(),
            )),
            mask.decl(sub(len(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    entry.decl(idx(index.e(), slot.e(), i64())),
                    when(lt(entry.e(), int(0)), vec![ret(int(-1))]),
                    when(
                        matches(at(d.e(), add(mul(entry.e(), int(2)), int(1))), key.e()),
                        vec![ret(add(mul(entry.e(), int(2)), int(1)))],
                    ),
                    slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    let mixed = |k: Expr| call("zb_hash_mix", vec![hash_of(k)], i64());
    out.push(define(
        &name("find"),
        &[&d, &key],
        i64(),
        vec![
            when(
                eq(at(d.e(), int(0)), null(any())),
                vec![
                    n.decl(len(d.e())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            when(matches(at(d.e(), i.e()), key.e()), vec![ret(i.e())]),
                            i.add_assign(int(2)),
                        ],
                    ),
                    ret(int(-1)),
                ],
            ),
            ret(call(
                &name("find_hashed"),
                vec![d.e(), key.e(), mixed(key.e())],
                i64(),
            )),
        ],
    ));
    let find = || call(&name("find"), vec![d.e(), key.e()], i64());
    out.push(define(
        &name("contains"),
        &[&d, &key],
        boolean(),
        vec![ret(ge(find(), int(0)))],
    ));
    out.push(define(
        &name("get"),
        &[&d, &key],
        any(),
        vec![
            i.decl(find()),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", text(key.e()))]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out.push(define(
        &name("get_default"),
        &[&d, &key, &default],
        any(),
        vec![
            i.decl(find()),
            when(lt(i.e(), int(0)), vec![ret(default.e())]),
            ret(at(d.e(), add(i.e(), int(1)))),
        ],
    ));
    out.push(define(
        &name("set"),
        &[&d, &key, &v],
        unit(),
        vec![
            when(
                eq(at(d.e(), int(0)), null(any())),
                vec![
                    i.decl(find()),
                    if_(
                        lt(i.e(), int(0)),
                        vec![expr(call(
                            "zb_dict_insert",
                            vec![d.e(), boxed(key.e()), v.e()],
                            unit(),
                        ))],
                        vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(mixed(key.e())),
            i.decl(call(
                &name("find_hashed"),
                vec![d.e(), key.e(), h.e()],
                i64(),
            )),
            if_(
                lt(i.e(), int(0)),
                vec![expr(call(
                    "zb_dict_insert_hashed",
                    vec![d.e(), boxed(key.e()), v.e(), h.e()],
                    unit(),
                ))],
                vec![set_idx(d.e(), add(i.e(), int(1)), v.e())],
            ),
            ret_void(),
        ],
    ));
    out
}

/// `name(s, key) -> bool`: whether a set holds a value matching `key`,
/// a value of `key_ty` that is not a box. `hash_of` is the key's hash
/// as [`zb_any_hash`] would hash its boxed form, so the probe lands on
/// the same slot; `matches(stored, key)` compares a stored box with it.
/// A set below its table size is scanned instead.
pub(crate) fn set_contains_by(
    name: &str,
    list_type: TypeId,
    key_ty: Type,
    hash_of: &dyn Fn(Expr) -> Expr,
    matches: &dyn Fn(Expr, Expr) -> Expr,
) -> Decl {
    let anys = list_of(list_type, any());
    let ints = list_of(list_type, i64());
    let s = borrowed("s", anys.clone());
    let key = borrowed("key", key_ty);
    let n = local("n", i64());
    let i = local("i", i64());
    let index = local("index", ints.clone());
    let mask = local("mask", i64());
    let slot = local("slot", i64());
    let entry = local("e", i64());
    let h = local("h", i64());
    let mut scan = vec![n.decl(len(s.e()))];
    scan.extend(for_range(
        &i,
        int(1),
        n.e(),
        vec![when(
            matches(at(s.e(), i.e()), key.e()),
            vec![ret(bool(true))],
        )],
    ));
    scan.push(ret(bool(false)));
    define(
        name,
        &[&s, &key],
        boolean(),
        vec![
            when(eq(at(s.e(), int(0)), null(any())), scan),
            index.decl(call(
                "zb_unbox_list_raw_i64",
                vec![at(s.e(), int(0))],
                ints.clone(),
            )),
            mask.decl(sub(div(sub(len(index.e()), int(1)), int(2)), int(1))),
            h.decl(call("zb_hash_mix", vec![hash_of(key.e())], i64())),
            slot.decl(bitand(h.e(), mask.e())),
            i.decl(int(0)),
            while_(
                le(i.e(), mask.e()),
                vec![
                    entry.decl(idx(index.e(), add(mul(slot.e(), int(2)), int(1)), i64())),
                    when(lt(entry.e(), int(0)), vec![ret(bool(false))]),
                    when(
                        and(
                            eq(
                                idx(index.e(), add(mul(slot.e(), int(2)), int(2)), i64()),
                                h.e(),
                            ),
                            matches(at(s.e(), add(entry.e(), int(1))), key.e()),
                        ),
                        vec![ret(bool(true))],
                    ),
                    slot.set(bitand(add(slot.e(), int(1)), mask.e())),
                    i.add_assign(int(1)),
                ],
            ),
            ret(bool(false)),
        ],
    )
}

fn set(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let ints = list_of(list_type, i64());
    // Every function edits the set in place; only boxing keeps it.
    let s = borrowed("s", anys.clone());
    let other = borrowed("other", anys.clone());
    let v = kept("v", any());
    // Lookups and removals read the value and keep nothing.
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
    let index = borrowed("index", ints.clone());
    let slot = local("slot", i64());
    let entry = local("e", i64());
    let cap = local("cap", i64());
    let h = local("h", i64());
    let kcat = local("kcat", i64());
    let stored = local("stored", any());
    let mut d = Vec::new();

    // A set is laid out as a dict is: position 0 holds the hash index
    // (None while the set is small), the values follow in insertion
    // order, so value `e` is at position 1 + e. The table's first word
    // caches the set's small-int mask (MASK_UNKNOWN until asked for,
    // -1 when the values are not all small ints); then two words per
    // slot, the entry number (-1 when empty) and the entry's hash, so
    // growing it and copying it hash nothing again and a probe reads a
    // value only when its hash agrees.
    let count = |s: Expr| sub(len(s), int(1));
    let value_at = |s: Expr, e: Expr| at(s, add(e, int(1)));
    let index_of = |s: Expr| call("zb_unbox_list_raw_i64", vec![at(s, int(0))], ints.clone());
    let box_index = |index: Expr| call("zb_list_box_i64", vec![index], any());
    let entry_at = |index: Expr, slot: Expr| idx(index, add(mul(slot, int(2)), int(1)), i64());
    let hash_at = |index: Expr, slot: Expr| idx(index, add(mul(slot, int(2)), int(2)), i64());
    let slots_of = |index: Expr| div(sub(len(index), int(1)), int(2));
    /// The cached mask word before it is computed.
    const MASK_UNKNOWN: i64 = -2;
    let forget_mask = |index: Expr| set_idx(index, int(0), int(MASK_UNKNOWN));
    let hash = |k: Expr| call("zb_dict_hash", vec![k], i64());
    let next_slot = |s: Expr, mask: Expr| bitand(add(s, int(1)), mask);
    let unindexed = |s: Expr| eq(at(s, int(0)), null(any()));
    let reuse = local("reuse", boolean());
    let old = local("old", ints.clone());
    let j = local("j", i64());
    let contains = |s: Expr, v: Expr| call("zb_set_contains", vec![s, v], boolean());
    let find = |s: Expr, v: Expr| call("zb_set_find", vec![s, v], i64());
    /// Values a set holds before it takes a table.
    const SMALL: i64 = 8;
    /// The first table's size, for a set just past `SMALL`.
    const FIRST_TABLE: i64 = 32;

    d.push(define(
        "zb_set_new",
        &[],
        anys.clone(),
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            push(out.e(), null(any())),
            ret(out.e()),
        ],
    ));
    d.push(define("zb_set_len", &[&s], i64(), vec![ret(count(s.e()))]));
    // An empty table of `cap` slots: every word -1, which is the empty
    // entry mark, and a hash no probe reads.
    d.push(define("zb_set_index_new", &[&cap], ints.clone(), {
        let table = local("table", ints.clone());
        vec![
            table.decl(list(Vec::new(), ints.clone())),
            expr(mcall(
                table.e(),
                "resize_filled",
                vec![add(mul(cap.e(), int(2)), int(1)), int(0xFF)],
                unit(),
            )),
            forget_mask(table.e()),
            ret(table.e()),
        ]
    }));
    // Record entry `e`, whose hash is `h`, in `index`, which has room.
    d.push(define(
        "zb_set_place_hashed",
        &[&index, &entry, &h],
        unit(),
        vec![
            mask.decl(sub(slots_of(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            while_(
                ge(entry_at(index.e(), slot.e()), int(0)),
                vec![slot.set(next_slot(slot.e(), mask.e()))],
            ),
            set_idx(index.e(), add(mul(slot.e(), int(2)), int(1)), entry.e()),
            set_idx(index.e(), add(mul(slot.e(), int(2)), int(2)), h.e()),
            ret_void(),
        ],
    ));
    // A fresh table of `cap` slots over every value of `s`: from the
    // hashes the old table holds when the entries kept their positions
    // (`reuse`), else from the values.
    d.push(define("zb_set_reindex", &[&s, &cap, &reuse], unit(), {
        let mut from_values = vec![n.decl(count(s.e()))];
        from_values.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_place_hashed",
                vec![index.e(), i.e(), hash(value_at(s.e(), i.e()))],
                unit(),
            ))],
        ));
        let mut from_table = vec![old.decl(index_of(s.e())), n.decl(slots_of(old.e()))];
        from_table.extend(for_range(
            &j,
            int(0),
            n.e(),
            vec![
                entry.decl(entry_at(old.e(), j.e())),
                when(
                    ge(entry.e(), int(0)),
                    vec![expr(call(
                        "zb_set_place_hashed",
                        vec![index.e(), entry.e(), hash_at(old.e(), j.e())],
                        unit(),
                    ))],
                ),
            ],
        ));
        vec![
            index.decl(call("zb_set_index_new", vec![cap.e()], ints.clone())),
            if_(
                and(reuse.e(), not(unindexed(s.e()))),
                from_table,
                from_values,
            ),
            set_idx(s.e(), int(0), box_index(index.e())),
            ret_void(),
        ]
    }));
    // A set built by appending distinct values takes its table once it
    // is past a few of them.
    d.push(define(
        "zb_set_settle",
        &[&s],
        unit(),
        vec![
            when(
                and(unindexed(s.e()), gt(count(s.e()), int(SMALL))),
                vec![
                    cap.decl(int(FIRST_TABLE)),
                    while_(
                        lt(cap.e(), mul(count(s.e()), int(2))),
                        vec![cap.set(mul(cap.e(), int(2)))],
                    ),
                    expr(call(
                        "zb_set_reindex",
                        vec![s.e(), cap.e(), bool(false)],
                        unit(),
                    )),
                ],
            ),
            ret_void(),
        ],
    ));
    // The position of `v` in a set with a table, given its hash, or -1;
    // and its position in any set, or -1.
    d.push(define(
        "zb_set_find_hashed",
        &[&s, &value, &h],
        i64(),
        vec![
            index.decl(index_of(s.e())),
            mask.decl(sub(slots_of(index.e()), int(1))),
            slot.decl(bitand(h.e(), mask.e())),
            kcat.decl(call("zb_any_category", vec![v.e()], i64())),
            i.decl(int(0)),
            while_(le(i.e(), mask.e()), {
                let mut matched = vec![stored.decl(value_at(s.e(), entry.e()))];
                matched.extend(when_key_matches(&stored, &v, &kcat, add(entry.e(), int(1))));
                vec![
                    entry.decl(entry_at(index.e(), slot.e())),
                    when(lt(entry.e(), int(0)), vec![ret(int(-1))]),
                    when(eq(hash_at(index.e(), slot.e()), h.e()), matched),
                    slot.set(next_slot(slot.e(), mask.e())),
                    i.add_assign(int(1)),
                ]
            }),
            ret(int(-1)),
        ],
    ));
    d.push(define(
        "zb_set_find",
        &[&s, &value],
        i64(),
        vec![
            when(
                unindexed(s.e()),
                vec![
                    n.decl(len(s.e())),
                    i.decl(int(1)),
                    while_(
                        lt(i.e(), n.e()),
                        vec![
                            when(key_matches(at(s.e(), i.e()), v.e()), vec![ret(i.e())]),
                            i.add_assign(int(1)),
                        ],
                    ),
                    ret(int(-1)),
                ],
            ),
            ret(call(
                "zb_set_find_hashed",
                vec![s.e(), v.e(), hash(v.e())],
                i64(),
            )),
        ],
    ));
    d.push(define(
        "zb_set_contains",
        &[&s, &value],
        boolean(),
        vec![ret(ge(find(s.e(), v.e()), int(0)))],
    ));
    // Add a value known to be absent whose hash is `h`, growing the
    // table first when the value would bring it to half full.
    d.push(define(
        "zb_set_insert_hashed",
        &[&s, &v, &h],
        unit(),
        vec![
            entry.decl(count(s.e())),
            cap.decl(slots_of(index_of(s.e()))),
            when(
                gt(mul(add(entry.e(), int(1)), int(2)), cap.e()),
                vec![expr(call(
                    "zb_set_reindex",
                    vec![s.e(), mul(cap.e(), int(2)), bool(true)],
                    unit(),
                ))],
            ),
            push(s.e(), v.e()),
            expr(call(
                "zb_set_place_hashed",
                vec![index_of(s.e()), entry.e(), h.e()],
                unit(),
            )),
            forget_mask(index_of(s.e())),
            ret_void(),
        ],
    ));
    // The hash of each value, by entry: read off the table when the set
    // has one, else computed; a set past `SMALL` values always has one.
    let hs = local("hs", ints.clone());
    d.push(define("zb_set_entry_hashes", &[&s], ints.clone(), {
        let from_values = for_range(
            &i,
            int(0),
            n.e(),
            vec![set_idx(hs.e(), i.e(), hash(value_at(s.e(), i.e())))],
        );
        let mut from_table = vec![index.decl(index_of(s.e())), cap.decl(slots_of(index.e()))];
        from_table.extend(for_range(
            &j,
            int(0),
            cap.e(),
            vec![
                entry.decl(entry_at(index.e(), j.e())),
                when(
                    ge(entry.e(), int(0)),
                    vec![set_idx(hs.e(), entry.e(), hash_at(index.e(), j.e()))],
                ),
            ],
        ));
        vec![
            n.decl(count(s.e())),
            hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(hs.e(), "resize_filled", vec![n.e(), int(0)], unit())),
            if_(unindexed(s.e()), from_values, from_table),
            ret(hs.e()),
        ]
    }));
    // Whether `v`, whose hash is `h`, is in `s`: by the table when there
    // is one, else by the short scan.
    d.push(define(
        "zb_set_has_hashed",
        &[&s, &value, &h],
        boolean(),
        vec![
            when(unindexed(s.e()), vec![ret(ge(find(s.e(), v.e()), int(0)))]),
            ret(ge(
                call("zb_set_find_hashed", vec![s.e(), v.e(), h.e()], i64()),
                int(0),
            )),
        ],
    ));
    // A set built from distinct values whose hashes are known: the
    // table, when the count asks for one, is filled from those hashes.
    let out_hs = borrowed("out_hs", ints.clone());
    d.push(define("zb_set_settle_hashed", &[&s, &out_hs], unit(), {
        let mut place = vec![
            cap.decl(int(FIRST_TABLE)),
            while_(
                lt(cap.e(), mul(n.e(), int(2))),
                vec![cap.set(mul(cap.e(), int(2)))],
            ),
            index.decl(call("zb_set_index_new", vec![cap.e()], ints.clone())),
        ];
        place.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(call(
                "zb_set_place_hashed",
                vec![index.e(), i.e(), idx(out_hs.e(), i.e(), i64())],
                unit(),
            ))],
        ));
        place.push(set_idx(s.e(), int(0), box_index(index.e())));
        vec![
            n.decl(count(s.e())),
            when(gt(n.e(), int(SMALL)), place),
            ret_void(),
        ]
    }));
    d.push(define(
        "zb_set_add",
        &[&s, &v],
        unit(),
        vec![
            when(
                unindexed(s.e()),
                vec![
                    when(
                        lt(find(s.e(), v.e()), int(0)),
                        vec![
                            push(s.e(), v.e()),
                            when(
                                gt(count(s.e()), int(SMALL)),
                                vec![expr(call(
                                    "zb_set_reindex",
                                    vec![s.e(), int(FIRST_TABLE), bool(false)],
                                    unit(),
                                ))],
                            ),
                        ],
                    ),
                    ret_void(),
                ],
            ),
            h.decl(hash(v.e())),
            when(
                lt(
                    call("zb_set_find_hashed", vec![s.e(), v.e(), h.e()], i64()),
                    int(0),
                ),
                vec![expr(call(
                    "zb_set_insert_hashed",
                    vec![s.e(), v.e(), h.e()],
                    unit(),
                ))],
            ),
            ret_void(),
        ],
    ));
    // Removal shifts the later values down, so the table is rebuilt.
    let remove_at = |i: &Local| {
        vec![
            expr(mcall(s.e(), "remove_at", vec![i.e()], any())),
            when(
                not(unindexed(s.e())),
                vec![expr(call(
                    "zb_set_reindex",
                    vec![s.e(), slots_of(index_of(s.e())), bool(false)],
                    unit(),
                ))],
            ),
        ]
    };
    d.push(define("zb_set_discard", &[&s, &value], unit(), {
        let mut st = vec![i.decl(find(s.e(), v.e()))];
        st.push(when(ge(i.e(), int(0)), remove_at(&i)));
        st.push(ret_void());
        st
    }));
    d.push(define("zb_set_remove", &[&s, &value], unit(), {
        let mut st = vec![
            i.decl(find(s.e(), v.e())),
            when(lt(i.e(), int(0)), vec![fatal("KeyError", any_str(v.e()))]),
        ];
        st.extend(remove_at(&i));
        st.push(ret_void());
        st
    }));
    d.push(define(
        "zb_set_clear",
        &[&s],
        unit(),
        vec![
            expr(mcall(s.e(), "clear", vec![], unit())),
            push(s.e(), null(any())),
            ret_void(),
        ],
    ));
    // The values, in order, as a list of their own.
    d.push(define("zb_set_items", &[&s], anys.clone(), {
        vec![
            out.decl(list(Vec::new(), anys.clone())),
            n.decl(len(s.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![push(out.e(), at(s.e(), i.e())), i.add_assign(int(1))],
            ),
            ret(out.e()),
        ]
    }));
    // A copy with a table of its own: the values keep their positions,
    // so the table is copied rather than built again.
    d.push(define(
        "zb_set_copy",
        &[&s],
        anys.clone(),
        vec![
            out.decl(call("zb_list_copy_any", vec![s.e()], anys.clone())),
            when(
                not(unindexed(out.e())),
                vec![set_idx(
                    out.e(),
                    int(0),
                    box_index(call(
                        "zb_list_copy_i64",
                        vec![index_of(s.e())],
                        ints.clone(),
                    )),
                )],
            ),
            ret(out.e()),
        ],
    ));
    // A set from any list, keeping first occurrences.
    let xs = local("xs", anys.clone());
    d.push(define("zb_set_from", &[&xs], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
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
    // The hash of a set is the same whatever order it was filled in.
    d.push(define("zb_set_hash", &[&s], i64(), {
        vec![
            h.decl(int(0x2545_F491_4F6C_DD1D)),
            n.decl(len(s.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    h.set(bitxor(h.e(), hash(at(s.e(), i.e())))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(add(mul(h.e(), int(1_000_003)), count(s.e()))),
        ]
    }));

    // Integer board positions fit in a word. -1 means the set also
    // holds a value that needs the general equality path.
    // Computed by a scan of the values; a set with a table keeps the
    // answer in the table's first word until a value is added.
    d.push(define("zb_set_mask63_scan", &[&s], i64(), {
        let mut st = vec![mask.decl(int(0)), n.decl(len(s.e()))];
        st.extend(for_range(
            &i,
            int(1),
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
                bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
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
    d.push(define(
        "zb_set_mask63",
        &[&s],
        i64(),
        vec![
            when(
                unindexed(s.e()),
                vec![ret(call("zb_set_mask63_scan", vec![s.e()], i64()))],
            ),
            index.decl(index_of(s.e())),
            mask.decl(idx(index.e(), int(0), i64())),
            when(ne(mask.e(), int(MASK_UNKNOWN)), vec![ret(mask.e())]),
            mask.set(call("zb_set_mask63_scan", vec![s.e()], i64())),
            set_idx(index.e(), int(0), mask.e()),
            ret(mask.e()),
        ],
    ));
    d.push(define("zb_set_all_kind", &[&s, &wanted], boolean(), {
        let mut st = vec![
            n.decl(len(s.e())),
            when(eq(n.e(), int(1)), vec![ret(bool(false))]),
        ];
        st.extend(for_range(
            &i,
            int(1),
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
            // Two sets whose first values are of one kind share it, and
            // an empty set has no kind to be all of.
            when(
                or(
                    or(eq(len(s.e()), int(1)), eq(len(other.e()), int(1))),
                    eq(
                        call("zb_any_kind", vec![at(s.e(), int(1))], i64()),
                        call("zb_any_kind", vec![at(other.e(), int(1))], i64()),
                    ),
                ),
                vec![ret(bool(false))],
            ),
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

    // Intersection, union, difference, symmetric difference. A result
    // built from values known to be distinct is appended to and takes
    // its table at the end.
    let settled = |out: &Local| {
        vec![
            expr(call("zb_set_settle", vec![out.e()], unit())),
            ret(out.e()),
        ]
    };
    d.push(define("zb_set_and", &[&s, &other], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut matched = for_range(
                    &i,
                    int(1),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
                        when(
                            ne(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                matched.extend(settled(&out));
                fast.push(when(ge(right_mask.e(), int(0)), matched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(out.e())],
            ),
            hs.decl(call("zb_set_entry_hashes", vec![s.e()], ints.clone())),
            out_hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
            expr(mcall(out_hs.e(), "reserve", vec![n.e()], unit())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    call(
                        "zb_set_has_hashed",
                        vec![other.e(), at(s.e(), i.e()), h.e()],
                        boolean(),
                    ),
                    vec![push(out.e(), at(s.e(), i.e())), push(out_hs.e(), h.e())],
                ),
            ],
        ));
        st.push(expr(call(
            "zb_set_settle_hashed",
            vec![out.e(), out_hs.e()],
            unit(),
        )));
        st.push(ret(out.e()));
        st
    }));
    // Room for `extra` more values without the table growing as they
    // come: a table sized once for what a union adds.
    let extra = local("extra", i64());
    let need = local("need", i64());
    d.push(define(
        "zb_set_reserve",
        &[&s, &extra],
        unit(),
        vec![
            need.decl(mul(add(count(s.e()), extra.e()), int(2))),
            when(
                unindexed(s.e()),
                vec![
                    when(
                        gt(add(count(s.e()), extra.e()), int(SMALL)),
                        vec![
                            cap.decl(int(FIRST_TABLE)),
                            while_(lt(cap.e(), need.e()), vec![cap.set(mul(cap.e(), int(2)))]),
                            expr(call(
                                "zb_set_reindex",
                                vec![s.e(), cap.e(), bool(false)],
                                unit(),
                            )),
                        ],
                    ),
                    ret_void(),
                ],
            ),
            cap.decl(slots_of(index_of(s.e()))),
            when(
                lt(cap.e(), need.e()),
                vec![
                    while_(lt(cap.e(), need.e()), vec![cap.set(mul(cap.e(), int(2)))]),
                    expr(call(
                        "zb_set_reindex",
                        vec![s.e(), cap.e(), bool(true)],
                        unit(),
                    )),
                ],
            ),
            ret_void(),
        ],
    ));
    // Every value of `other` added to `s`, by the hashes `other` holds:
    // into the table when `s` has one after making room, else by the
    // short path.
    d.push(define("zb_set_update", &[&s, &other], unit(), {
        let mut st = vec![
            expr(call(
                "zb_set_reserve",
                vec![s.e(), count(other.e())],
                unit(),
            )),
            n.decl(len(other.e())),
        ];
        let mut by_hash = vec![hs.decl(call("zb_set_entry_hashes", vec![other.e()], ints.clone()))];
        by_hash.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    lt(
                        call(
                            "zb_set_find_hashed",
                            vec![s.e(), at(other.e(), i.e()), h.e()],
                            i64(),
                        ),
                        int(0),
                    ),
                    vec![expr(call(
                        "zb_set_insert_hashed",
                        vec![s.e(), at(other.e(), i.e()), h.e()],
                        unit(),
                    ))],
                ),
            ],
        ));
        let mut by_add = Vec::new();
        by_add.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![expr(call(
                "zb_set_add",
                vec![s.e(), at(other.e(), i.e())],
                unit(),
            ))],
        ));
        st.push(if_(unindexed(s.e()), by_add, by_hash));
        st.push(ret_void());
        st
    }));
    d.push(define(
        "zb_set_or",
        &[&s, &other],
        anys.clone(),
        vec![
            out.decl(call("zb_set_copy", vec![s.e()], anys.clone())),
            expr(call("zb_set_update", vec![out.e(), other.e()], unit())),
            ret(out.e()),
        ],
    ));
    d.push(define("zb_set_sub", &[&s, &other], anys.clone(), {
        let mut st = vec![
            out.decl(call("zb_set_new", vec![], anys.clone())),
            n.decl(len(s.e())),
            left_mask.decl(call("zb_set_mask63", vec![s.e()], i64())),
            when(ge(left_mask.e(), int(0)), {
                let mut fast = vec![right_mask.decl(call("zb_set_mask63", vec![other.e()], i64()))];
                let mut unmatched = for_range(
                    &i,
                    int(1),
                    n.e(),
                    vec![
                        member.decl(at(s.e(), i.e())),
                        bit.decl(call("zb_box_payload_i64", vec![member.e()], i64())),
                        when(
                            eq(bitand(right_mask.e(), shl(int(1), bit.e())), int(0)),
                            vec![push(out.e(), member.e())],
                        ),
                    ],
                );
                unmatched.extend(settled(&out));
                fast.push(when(ge(right_mask.e(), int(0)), unmatched));
                fast
            }),
            when(
                call("zb_set_disjoint_kinds", vec![s.e(), other.e()], boolean()),
                vec![ret(call("zb_set_copy", vec![s.e()], anys.clone()))],
            ),
            hs.decl(call("zb_set_entry_hashes", vec![s.e()], ints.clone())),
            out_hs.decl(list(Vec::new(), ints.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
            expr(mcall(out_hs.e(), "reserve", vec![n.e()], unit())),
        ];
        st.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                h.decl(idx(hs.e(), sub(i.e(), int(1)), i64())),
                when(
                    not(call(
                        "zb_set_has_hashed",
                        vec![other.e(), at(s.e(), i.e()), h.e()],
                        boolean(),
                    )),
                    vec![push(out.e(), at(s.e(), i.e())), push(out_hs.e(), h.e())],
                ),
            ],
        ));
        st.push(expr(call(
            "zb_set_settle_hashed",
            vec![out.e(), out_hs.e()],
            unit(),
        )));
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
            when(gt(count(s.e()), count(other.e())), vec![ret(bool(false))]),
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
            int(1),
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
            eq(count(s.e()), count(other.e())),
            call("zb_set_issubset", vec![s.e(), other.e()], boolean()),
        ))],
    ));
    d.push(define(
        "zb_set_repr",
        &[&s],
        string(),
        vec![
            when(eq(count(s.e()), int(0)), vec![ret(text("set()"))]),
            text_out.decl(call(
                "zb_list_items_any",
                vec![
                    call("zb_set_items", vec![s.e()], anys.clone()),
                    text("{"),
                    text("}"),
                ],
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
    let boxed = local("s", anys.clone());
    d.push(define(
        "zb_set_box",
        &[&boxed],
        any(),
        vec![ret(call(
            "zb_box_set_raw",
            vec![boxed.e(), int32(SET_TAG as i32)],
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
    // The box itself, once it is known to hold a set: what a slot that
    // stores sets as boxes takes from a dynamic value.
    d.push(define(
        "zb_set_as_box",
        &[&x],
        any(),
        vec![
            expr(call("zb_set_unbox", vec![x.e()], anys.clone())),
            ret(x.e()),
        ],
    ));
    d
}
