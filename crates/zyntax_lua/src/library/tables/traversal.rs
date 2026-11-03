//! Traversal declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        arr,
        cat,
        h,
        i,
        k,
        n,
        tb,
        ..
    } = context;
    let arr_of = |tb: Expr| crate::library::arr_of(tb, t);
    let hash_of = |tb: Expr| crate::library::hash_of(tb, t);
    let shaped = |tb: Expr| crate::library::is_shaped(tb);
    let slot_index = |tb: Expr, name: Expr| call("zl_shape_index", vec![tb, name], i64());
    let slot_load = |tb: Expr, i: Expr| call("zl_shape_load", vec![tb, i], any());
    let slot_count = |tb: Expr| call("zl_shape_count", vec![tb], i64());
    let slot_key = |tb: Expr, i: Expr| call("zl_shape_key", vec![tb, i], string());
    let slot = local("slot", i64());

    // ─── traversal ──────────────────────────────────────────────
    // A traversal position: `0..n` for the array part, `n + j` for
    // slot `j` of a shaped table, `n + slots + e` for entry `e` of the
    // hash part. The next position at or after `pos` holding a value,
    // or -1 when the traversal is over.
    let pos = local("pos", i64());
    let count = local("count", i64());
    let slots = local("slots", i64());
    d.push(define(
        "zl_next_pos",
        &[tb, &pos],
        i64(),
        vec![
            arr.decl(arr_of(tb.e())),
            n.decl(len(arr.e())),
            i.decl(pos.e()),
            while_(
                lt(i.e(), n.e()),
                vec![
                    when(not(is_nil(at(arr.e(), i.e()))), vec![ret(i.e())]),
                    i.add_assign(int(1)),
                ],
            ),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            while_(
                lt(sub(i.e(), n.e()), slots.e()),
                vec![
                    when(
                        crate::library::slot_present(tb.e(), sub(i.e(), n.e())),
                        vec![ret(i.e())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            when(is_nil(hash_field(tb.e())), vec![ret(int(-1))]),
            h.decl(hash_of(tb.e())),
            count.decl(call("zb_dict_pair_count", vec![h.e()], i64())),
            while_(
                lt(sub(sub(i.e(), n.e()), slots.e()), count.e()),
                vec![
                    when(
                        not(is_nil(call(
                            "zb_dict_value_at",
                            vec![h.e(), sub(sub(i.e(), n.e()), slots.e())],
                            any(),
                        ))),
                        vec![ret(i.e())],
                    ),
                    i.add_assign(int(1)),
                ],
            ),
            ret(int(-1)),
        ],
    ));
    d.push(define(
        "zl_pos_key",
        &[tb, &pos],
        any(),
        vec![
            n.decl(len(arr_of(tb.e()))),
            when(lt(pos.e(), n.e()), vec![ret(box_i64(add(pos.e(), int(1))))]),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            when(
                lt(sub(pos.e(), n.e()), slots.e()),
                vec![ret(box_str(slot_key(tb.e(), sub(pos.e(), n.e()))))],
            ),
            ret(call(
                "zb_dict_key_at",
                vec![hash_of(tb.e()), sub(sub(pos.e(), n.e()), slots.e())],
                any(),
            )),
        ],
    ));
    d.push(define(
        "zl_pos_value",
        &[tb, &pos],
        any(),
        vec![
            arr.decl(arr_of(tb.e())),
            n.decl(len(arr.e())),
            when(lt(pos.e(), n.e()), vec![ret(at(arr.e(), pos.e()))]),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            when(
                lt(sub(pos.e(), n.e()), slots.e()),
                vec![ret(slot_load(tb.e(), sub(pos.e(), n.e())))],
            ),
            ret(call(
                "zb_dict_value_at",
                vec![hash_of(tb.e()), sub(sub(pos.e(), n.e()), slots.e())],
                any(),
            )),
        ],
    ));
    // The position just past key `k`: 0 for nil, the array slot for an
    // integer in the array part, the hash entry for anything else.
    d.push(define(
        "zl_pos_after",
        &[tb, k],
        i64(),
        vec![
            when(is_nil(k.e()), vec![ret(int(0))]),
            n.decl(len(arr_of(tb.e()))),
            cat.decl(category(k.e())),
            when(
                is_int_cat(cat),
                vec![
                    i.decl(get_i64(k.e())),
                    when(and(ge(i.e(), int(1)), le(i.e(), n.e())), vec![ret(i.e())]),
                ],
            ),
            slots.decl(if_expr(shaped(tb.e()), slot_count(tb.e()), int(0))),
            // A string naming a slot: the position after that slot.
            when(
                and(gt(slots.e(), int(0)), eq(cat.e(), int(STR))),
                vec![
                    slot.decl(slot_index(tb.e(), get_str(k.e()))),
                    when(
                        ge(slot.e(), int(0)),
                        vec![ret(add(add(n.e(), slot.e()), int(1)))],
                    ),
                ],
            ),
            when(
                not(is_nil(hash_field(tb.e()))),
                vec![
                    h.decl(hash_of(tb.e())),
                    i.decl(int(0)),
                    count.decl(call("zb_dict_pair_count", vec![h.e()], i64())),
                    while_(
                        lt(i.e(), count.e()),
                        vec![
                            when(
                                call(
                                    "zb_any_eq",
                                    vec![call("zb_dict_key_at", vec![h.e(), i.e()], any()), k.e()],
                                    boolean(),
                                ),
                                vec![ret(add(add(add(n.e(), slots.e()), i.e()), int(1)))],
                            ),
                            i.add_assign(int(1)),
                        ],
                    ),
                ],
            ),
            // An index past the array part but within its longest
            // was an element removed during the traversal, which
            // shortened the part: the traversal goes on from its end.
            when(
                and(
                    is_int_cat(cat),
                    and(
                        ge(get_i64(k.e()), int(1)),
                        le(get_i64(k.e()), crate::library::high_of(tb.e())),
                    ),
                ),
                vec![ret(n.e())],
            ),
            lua_error(text("invalid key to 'next'")),
            ret(int(-1)),
        ],
    ));
    // The position after `pos` when the array part held `seen` elements
    // at the last step: elements removed since moved what follows.
    let seen = local("seen", i64());
    d.push(define(
        "zl_next_pos_from",
        &[tb, &pos, &seen],
        i64(),
        vec![
            n.decl(len(arr_of(tb.e()))),
            i.decl(pos.e()),
            when(
                lt(n.e(), seen.e()),
                vec![if_(
                    ge(i.e(), seen.e()),
                    vec![i.set(sub(i.e(), sub(seen.e(), n.e())))],
                    vec![when(gt(i.e(), n.e()), vec![i.set(n.e())])],
                )],
            ),
            ret(call("zl_next_pos", vec![tb.e(), i.e()], i64())),
        ],
    ));
    // `next(t, k)`: the key and value after `k`, as two values, or
    // nothing.
    let found = local("found", any());
    d.push(define(
        "zl_next",
        &[tb, k],
        anys.clone(),
        vec![
            pos.decl(call(
                "zl_next_pos",
                vec![tb.e(), call("zl_pos_after", vec![tb.e(), k.e()], i64())],
                i64(),
            )),
            // The traversal ends with one nil.
            when(
                lt(pos.e(), int(0)),
                vec![ret(list(vec![nil()], anys.clone()))],
            ),
            // The value is read first and held: making the key may
            // collect, which clears a weak value nothing holds.
            found.decl(call("zl_pos_value", vec![tb.e(), pos.e()], any())),
            ret(list(
                vec![call("zl_pos_key", vec![tb.e(), pos.e()], any()), found.e()],
                anys.clone(),
            )),
        ],
    ));
}
