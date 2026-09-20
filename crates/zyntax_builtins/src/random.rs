//! Python's `random` module: the Mersenne Twister as CPython runs it, so
//! a seeded program draws the same numbers here as there. The state is
//! 624 words in a list the module holds boxed, with the index of the
//! next word; a first draw without a seed seeds from the clock.
//!
//! Words are 32-bit values in `i64`s, masked after each step; every
//! product here fits an `i64` because one factor is below 2^32 and the
//! other below 2^31.

use crate::build::*;
use crate::list_of;
use zyntax_typed_ast::TypeId;

const N: i64 = 624;
const M: i64 = 397;
const MASK32: i64 = 0xFFFF_FFFF;
const UPPER: i64 = 0x8000_0000;
const LOWER: i64 = 0x7FFF_FFFF;
const MATRIX_A: i64 = 0x9908_B0DF;

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let words = list_of(list_type, i64());
    let mut d = Vec::new();

    // The generator's state: the words, boxed, or the null box before
    // the first seed; the index of the next word to hand out.
    let state = local("zb_mt_state", any());
    let index = local("zb_mt_index", i64());
    d.push(global_var(state.name, any()));
    d.push(global_var(index.name, i64()));

    let mt = local("mt", words.clone());
    let i = local("i", i64());
    let j = local("j", i64());
    let k = local("k", i64());
    let y = local("y", i64());
    let s = local("s", i64());
    let mask = |e: Expr| bitand(e, int(MASK32));
    let at = |i: Expr| idx(mt.e(), i, i64());
    let put = |i: Expr, v: Expr| set_idx(mt.e(), i, v);
    let words_of_state = || call("zb_unbox_list_raw_i64", vec![state.e()], words.clone());

    // The state as a list: made at the first use.
    let mut make = vec![mt.decl(list(Vec::new(), words.clone()))];
    make.extend(for_range(
        &i,
        int(0),
        int(N),
        vec![expr(mcall(mt.e(), "push", vec![int(0)], unit()))],
    ));
    make.push(state.set(call("zb_list_box_i64", vec![mt.e()], any())));
    d.push(define(
        "zb_mt_words",
        &[],
        words.clone(),
        vec![
            when(eq(state.e(), null(any())), make),
            ret(words_of_state()),
        ],
    ));

    // init_genrand(s): the state from one 32-bit seed.
    d.push(define(
        "zb_mt_init_genrand",
        &[&s],
        unit(),
        vec![
            mt.decl(call("zb_mt_words", vec![], words.clone())),
            put(int(0), mask(s.e())),
            block_of(for_range(
                &i,
                int(1),
                int(N),
                vec![
                    y.decl(at(sub(i.e(), int(1)))),
                    put(
                        i.e(),
                        mask(add(
                            mul(int(1812433253), bitxor(y.e(), shr(y.e(), int(30)))),
                            i.e(),
                        )),
                    ),
                ],
            )),
            index.set(int(N)),
            ret_void(),
        ],
    ));

    // init_by_array(key): the state from a key of 32-bit words, as
    // `random.seed(int)` seeds from the int's words.
    let key = borrowed("key", words.clone());
    let key_len = local("key_len", i64());
    let rounds = local("rounds", i64());
    let prev = local("prev", i64());
    let wrap = || {
        when(
            ge(i.e(), int(N)),
            vec![put(int(0), at(int(N - 1))), i.set(int(1))],
        )
    };
    d.push(define(
        "zb_mt_init_by_array",
        &[&key],
        unit(),
        vec![
            expr(call("zb_mt_init_genrand", vec![int(19650218)], unit())),
            mt.decl(call("zb_mt_words", vec![], words.clone())),
            key_len.decl(mcall(key.e(), "len", vec![], i64())),
            i.decl(int(1)),
            j.decl(int(0)),
            rounds.decl(if_expr(gt(key_len.e(), int(N)), key_len.e(), int(N))),
            k.decl(rounds.e()),
            while_(
                gt(k.e(), int(0)),
                vec![
                    prev.decl(at(sub(i.e(), int(1)))),
                    put(
                        i.e(),
                        mask(add(
                            add(
                                bitxor(
                                    at(i.e()),
                                    mul(bitxor(prev.e(), shr(prev.e(), int(30))), int(1664525)),
                                ),
                                idx(key.e(), j.e(), i64()),
                            ),
                            j.e(),
                        )),
                    ),
                    i.add_assign(int(1)),
                    j.add_assign(int(1)),
                    wrap(),
                    when(ge(j.e(), key_len.e()), vec![j.set(int(0))]),
                    k.set(sub(k.e(), int(1))),
                ],
            ),
            k.set(int(N - 1)),
            while_(
                gt(k.e(), int(0)),
                vec![
                    prev.decl(at(sub(i.e(), int(1)))),
                    put(
                        i.e(),
                        mask(sub(
                            bitxor(
                                at(i.e()),
                                mul(bitxor(prev.e(), shr(prev.e(), int(30))), int(1566083941)),
                            ),
                            i.e(),
                        )),
                    ),
                    i.add_assign(int(1)),
                    wrap(),
                    k.set(sub(k.e(), int(1))),
                ],
            ),
            put(int(0), int(UPPER)),
            index.set(int(N)),
            ret_void(),
        ],
    ));

    // random.seed(n): the words of |n|, low first, as the key.
    let n = local("n", i64());
    let seed_words = local("seed_words", words.clone());
    d.push(define(
        "zb_random_seed",
        &[&n],
        unit(),
        vec![
            s.decl(if_expr(lt(n.e(), int(0)), neg(n.e()), n.e())),
            seed_words.decl(list(vec![mask(s.e())], words.clone())),
            when(
                gt(shr(s.e(), int(32)), int(0)),
                vec![expr(mcall(
                    seed_words.e(),
                    "push",
                    vec![mask(shr(s.e(), int(32)))],
                    unit(),
                ))],
            ),
            expr(call("zb_mt_init_by_array", vec![seed_words.e()], unit())),
            ret_void(),
        ],
    ));
    // random.seed(): from the clock.
    d.push(define(
        "zb_random_seed_clock",
        &[],
        unit(),
        vec![
            expr(call(
                "zb_random_seed",
                vec![cast(
                    mul(call("zb_time_time", vec![], f64()), float(1e6)),
                    i64(),
                )],
                unit(),
            )),
            ret_void(),
        ],
    ));

    // genrand_uint32(): the next word, the state regenerated every N.
    let kk = local("kk", i64());
    let regen = |kk_from: i64, kk_to: i64, next: i64, far: i64| {
        block_of(for_range(
            &kk,
            int(kk_from),
            int(kk_to),
            vec![
                y.decl(bitor(
                    bitand(at(kk.e()), int(UPPER)),
                    bitand(at(add(kk.e(), int(next))), int(LOWER)),
                )),
                put(
                    kk.e(),
                    bitxor(
                        bitxor(at(add(kk.e(), int(far))), shr(y.e(), int(1))),
                        mul(bitand(y.e(), int(1)), int(MATRIX_A)),
                    ),
                ),
            ],
        ))
    };
    d.push(define(
        "zb_mt_genrand",
        &[],
        i64(),
        vec![
            when(
                eq(state.e(), null(any())),
                vec![expr(call("zb_random_seed_clock", vec![], unit()))],
            ),
            mt.decl(words_of_state()),
            when(
                ge(index.e(), int(N)),
                vec![
                    regen(0, N - M, 1, M),
                    regen(N - M, N - 1, 1, M - N),
                    y.decl(bitor(
                        bitand(at(int(N - 1)), int(UPPER)),
                        bitand(at(int(0)), int(LOWER)),
                    )),
                    put(
                        int(N - 1),
                        bitxor(
                            bitxor(at(int(M - 1)), shr(y.e(), int(1))),
                            mul(bitand(y.e(), int(1)), int(MATRIX_A)),
                        ),
                    ),
                    index.set(int(0)),
                ],
            ),
            y.decl(at(index.e())),
            index.add_assign(int(1)),
            y.set(bitxor(y.e(), shr(y.e(), int(11)))),
            y.set(bitxor(y.e(), bitand(shl(y.e(), int(7)), int(0x9D2C_5680)))),
            y.set(bitxor(y.e(), bitand(shl(y.e(), int(15)), int(0xEFC6_0000)))),
            y.set(bitxor(y.e(), shr(y.e(), int(18)))),
            ret(mask(y.e())),
        ],
    ));

    // random.random(): 53 bits from two words.
    let a = local("a", i64());
    let b = local("b", i64());
    d.push(define(
        "zb_random_random",
        &[],
        f64(),
        vec![
            a.decl(shr(call("zb_mt_genrand", vec![], i64()), int(5))),
            b.decl(shr(call("zb_mt_genrand", vec![], i64()), int(6))),
            ret(mul(
                add(
                    mul(cast(a.e(), f64()), float(67108864.0)),
                    cast(b.e(), f64()),
                ),
                float(1.0 / 9007199254740992.0),
            )),
        ],
    ));

    // getrandbits(k) for 1 <= k <= 63: whole words low first, the last
    // one cut to the bits still owed.
    let bits = local("bits", i64());
    let r = local("r", i64());
    d.push(define(
        "zb_random_getrandbits",
        &[&bits],
        i64(),
        vec![
            when(
                le(bits.e(), int(32)),
                vec![ret(shr(
                    call("zb_mt_genrand", vec![], i64()),
                    sub(int(32), bits.e()),
                ))],
            ),
            r.decl(call("zb_mt_genrand", vec![], i64())),
            k.decl(sub(bits.e(), int(32))),
            ret(bitor(
                r.e(),
                shl(
                    shr(call("zb_mt_genrand", vec![], i64()), sub(int(32), k.e())),
                    int(32),
                ),
            )),
        ],
    ));

    // _randbelow(n): a draw of n.bit_length() bits until one is below n.
    let below = local("below", i64());
    d.push(define(
        "zb_random_below",
        &[&n],
        i64(),
        vec![
            bits.decl(int(0)),
            k.decl(n.e()),
            while_(
                gt(k.e(), int(0)),
                vec![bits.add_assign(int(1)), k.set(shr(k.e(), int(1)))],
            ),
            below.decl(call("zb_random_getrandbits", vec![bits.e()], i64())),
            while_(
                ge(below.e(), n.e()),
                vec![below.set(call("zb_random_getrandbits", vec![bits.e()], i64()))],
            ),
            ret(below.e()),
        ],
    ));

    // randrange(stop), randrange(start, stop), randrange(start, stop, step).
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let width = local("width", i64());
    d.push(define(
        "zb_random_randrange",
        &[&stop],
        i64(),
        vec![
            when(
                le(stop.e(), int(0)),
                vec![fatal("ValueError", text("empty range for randrange()"))],
            ),
            ret(call("zb_random_below", vec![stop.e()], i64())),
        ],
    ));
    d.push(define(
        "zb_random_randrange2",
        &[&start, &stop],
        i64(),
        vec![
            width.decl(sub(stop.e(), start.e())),
            when(
                le(width.e(), int(0)),
                vec![fatal("ValueError", text("empty range for randrange()"))],
            ),
            ret(add(
                start.e(),
                call("zb_random_below", vec![width.e()], i64()),
            )),
        ],
    ));
    d.push(define(
        "zb_random_randrange3",
        &[&start, &stop, &step],
        i64(),
        vec![
            width.decl(sub(stop.e(), start.e())),
            when(
                eq(step.e(), int(0)),
                vec![fatal("ValueError", text("zero step for randrange()"))],
            ),
            n.decl(if_expr(
                gt(step.e(), int(0)),
                div(add(width.e(), sub(step.e(), int(1))), step.e()),
                div(add(width.e(), add(step.e(), int(1))), step.e()),
            )),
            when(
                le(n.e(), int(0)),
                vec![fatal("ValueError", text("empty range for randrange()"))],
            ),
            ret(add(
                start.e(),
                mul(step.e(), call("zb_random_below", vec![n.e()], i64())),
            )),
        ],
    ));
    // randint(a, b) = randrange(a, b + 1).
    d.push(define(
        "zb_random_randint",
        &[&start, &stop],
        i64(),
        vec![ret(call(
            "zb_random_randrange2",
            vec![start.e(), add(stop.e(), int(1))],
            i64(),
        ))],
    ));
    // uniform(a, b) = a + (b - a) * random().
    let lo = local("lo", f64());
    let hi = local("hi", f64());
    d.push(define(
        "zb_random_uniform",
        &[&lo, &hi],
        f64(),
        vec![ret(add(
            lo.e(),
            mul(sub(hi.e(), lo.e()), call("zb_random_random", vec![], f64())),
        ))],
    ));
    // choice(seq): seq[_randbelow(len(seq))] over dynamic values.
    let anys = list_of(list_type, any());
    let seq = borrowed("seq", anys.clone());
    d.push(define(
        "zb_random_choice",
        &[&seq],
        any(),
        vec![
            n.decl(mcall(seq.e(), "len", vec![], i64())),
            when(
                eq(n.e(), int(0)),
                vec![fatal(
                    "IndexError",
                    text("Cannot choose from an empty sequence"),
                )],
            ),
            ret(idx(
                seq.e(),
                call("zb_random_below", vec![n.e()], i64()),
                any(),
            )),
        ],
    ));
    // shuffle(x): Fisher-Yates from the top, as CPython.
    let tmp = local("tmp", any());
    d.push(define(
        "zb_random_shuffle",
        &[&seq],
        unit(),
        vec![
            i.decl(sub(mcall(seq.e(), "len", vec![], i64()), int(1))),
            while_(
                gt(i.e(), int(0)),
                vec![
                    j.decl(call("zb_random_below", vec![add(i.e(), int(1))], i64())),
                    tmp.decl(idx(seq.e(), i.e(), any())),
                    set_idx(seq.e(), i.e(), idx(seq.e(), j.e(), any())),
                    set_idx(seq.e(), j.e(), tmp.e()),
                    i.set(sub(i.e(), int(1))),
                ],
            ),
            ret_void(),
        ],
    ));
    d
}
