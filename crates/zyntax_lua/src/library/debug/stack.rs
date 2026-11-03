//! Stack declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        acc,
        anys,
        event,
        f,
        i,
        k,
        key,
        line,
        meta,
        n,
        short,
        site,
        source,
        table,
        x,
        y,
        ..
    } = context;
    let site_now = || read_global(DBG_SITE, i64());
    let line_now = || read_global(LINE, i64());
    let mask = || read_global(HOOK_MASK, i64());
    let has_event = |bit: i64| ne(bitand(mask(), int(bit)), int(0));
    let table_of = |global: &'static str| {
        vec![when(
            is_nil(read_global(global, any())),
            vec![set_global(
                global,
                box_table(call("zl_table_new", vec![], table.clone())),
            )],
        )]
    };

    // ─── the stack ──────────────────────────────────────────────
    // A chunk's functions and call sites, and the accessor of its
    // functions' upvalues.
    d.push(define(
        "zl_dbg_register",
        &[k, source, short, meta, acc],
        unit(),
        {
            let mut st = vec![set_global(DBG_ON, bool(true))];
            st.push(expr(call(
                "zl_dbg_chunk_raw",
                vec![k.e(), source.e(), short.e(), meta.e()],
                unit(),
            )));
            st.extend(table_of(UPS));
            st.push(expr(call(
                "zl_rawseti",
                vec![unbox_table(read_global(UPS, any()), t), k.e(), acc.e()],
                unit(),
            )));
            st.push(ret_void());
            st
        },
    ));
    // The value a function of the program was made as, kept for
    // `getinfo(level, "f")`.
    d.push(define("zl_dbg_closure", &[key, f], unit(), {
        let mut st = table_of(FUNCS);
        st.push(expr(call(
            "zl_rawseti",
            vec![unbox_table(read_global(FUNCS, any()), t), key.e(), f.e()],
            unit(),
        )));
        st.push(ret_void());
        st
    }));
    // A hook told of `event`, with `x` (the line, or nil); no hook
    // runs inside another.
    d.push(define_cold(
        "zl_dbg_hook",
        &[event, x],
        unit(),
        vec![
            when(
                or(
                    read_global(HOOK_BUSY, boolean()),
                    is_nil(read_global(HOOK_FN, any())),
                ),
                vec![ret_void()],
            ),
            set_global(HOOK_BUSY, bool(true)),
            line.decl(line_now()),
            set_global(DBG_SITE, int(HOOK_SITE)),
            expr(call(
                "zl_call_2",
                vec![read_global(HOOK_FN, any()), box_str(event.e()), x.e()],
                any(),
            )),
            set_global(LINE, line.e()),
            set_global(HOOK_BUSY, bool(false)),
            ret_void(),
        ],
    ));
    // A function of the program entered: its frame, then the call hook.
    d.push(define(
        "zl_dbg_enter",
        &[key],
        unit(),
        vec![
            site.decl(site_now()),
            set_global(DBG_SITE, int(0)),
            expr(call(
                "zl_dbg_push",
                vec![key.e(), line_now(), site.e()],
                unit(),
            )),
            when(
                has_event(HOOK_CALL),
                vec![if_(
                    and(
                        ge(site.e(), int(0)),
                        ne(bitand(site.e(), int(TAIL_SITE)), int(0)),
                    ),
                    vec![expr(call(
                        "zl_dbg_hook",
                        vec![text("tail call"), nil()],
                        unit(),
                    ))],
                    vec![expr(call("zl_dbg_hook", vec![text("call"), nil()], unit()))],
                )],
            ),
            ret_void(),
        ],
    ));
    // A call into the library that may call the program back: a frame
    // with no function of the program's.
    d.push(define(
        "zl_dbg_enter_c",
        &[site],
        unit(),
        vec![
            set_global(DBG_SITE, int(0)),
            expr(call(
                "zl_dbg_push",
                vec![int(-1), line_now(), site.e()],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // A frame left: the return hook (not on an error's way out), then
    // the caller's line again.
    d.push(define(
        "zl_dbg_leave",
        &[],
        unit(),
        vec![
            when(
                and(has_event(HOOK_RETURN), is_nil(pending())),
                vec![expr(call(
                    "zl_dbg_hook",
                    vec![text("return"), nil()],
                    unit(),
                ))],
            ),
            set_global(LINE, call("zl_dbg_pop", vec![], i64())),
            ret_void(),
        ],
    ));
    // A statement at `line` starts, with a hook set: the count hook
    // every so many, the line hook at each new line.
    d.push(define(
        "zl_dbg_line",
        &[line],
        unit(),
        vec![
            when(
                has_event(HOOK_COUNT),
                vec![
                    set_global(HOOK_LEFT, sub(read_global(HOOK_LEFT, i64()), int(1))),
                    when(
                        le(read_global(HOOK_LEFT, i64()), int(0)),
                        vec![
                            set_global(HOOK_LEFT, read_global(HOOK_COUNT_EVERY, i64())),
                            expr(call("zl_dbg_hook", vec![text("count"), nil()], unit())),
                        ],
                    ),
                ],
            ),
            when(
                and(
                    has_event(HOOK_LINE),
                    call("zl_dbg_line_raw", vec![line.e(), bool(false)], boolean()),
                ),
                vec![expr(call(
                    "zl_dbg_hook",
                    vec![text("line"), hook_line(line.e())],
                    unit(),
                ))],
            ),
            ret_void(),
        ],
    ));
    // A loop goes back: its next statement is a line event whatever
    // its line.
    d.push(define(
        "zl_dbg_back",
        &[],
        unit(),
        vec![
            when(
                has_event(HOOK_LINE),
                vec![expr(call("zl_dbg_back_raw", vec![], unit()))],
            ),
            ret_void(),
        ],
    ));
    // A loop's body ended and its head runs on `line`, then jumps back
    // into the body, whose next statement is a line event whatever its
    // line.
    let line_event = |line: Expr, back: bool| {
        when(
            and(
                has_event(HOOK_LINE),
                call("zl_dbg_line_raw", vec![line.clone(), bool(back)], boolean()),
            ),
            vec![expr(call(
                "zl_dbg_hook",
                vec![text("line"), hook_line(line)],
                unit(),
            ))],
        )
    };
    d.push(define(
        "zl_dbg_loop",
        &[line],
        unit(),
        vec![
            line_event(line.e(), false),
            when(
                has_event(HOOK_LINE),
                vec![expr(call("zl_dbg_back_raw", vec![], unit()))],
            ),
            ret_void(),
        ],
    ));
    // A `while` jumps back to its test on `line`: a line event.
    d.push(define(
        "zl_dbg_loop_head",
        &[line],
        unit(),
        vec![line_event(line.e(), true), ret_void()],
    ));
    // A loop whose head is on `line` is left: the next statement is a
    // line event only on a line of its own.
    d.push(define(
        "zl_dbg_loop_exit",
        &[line],
        unit(),
        vec![
            when(
                has_event(HOOK_LINE),
                vec![expr(call("zl_dbg_set_line_raw", vec![line.e()], unit()))],
            ),
            ret_void(),
        ],
    ));
    // Slot `i` of the spill list `x` when `debug.setlocal` wrote it
    // (bit `i` of `n`), else the local's value `y` as the call left it.
    d.push(define(
        "zl_dbg_pick",
        &[x, n, i, y],
        any(),
        vec![
            when(
                ne(bitand(n.e(), shl(int(1), i.e())), int(0)),
                vec![ret(at(
                    call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                    i.e(),
                ))],
            ),
            ret(y.e()),
        ],
    ));
    // The running thread's hook as a value: a list of the hook, its
    // mask, its count and what is left of the count; nil for none.
    // A mask without a function is a hook inherited from the thread's
    // creator: `gethook` reports it, nothing calls it.
    d.push(define(
        "zl_dbg_hook_pack",
        &[],
        any(),
        vec![
            when(
                and(
                    is_nil(read_global(HOOK_FN, any())),
                    eq(read_global(HOOK_MASK, i64()), int(0)),
                ),
                vec![ret(nil())],
            ),
            ret(call(
                "zb_list_box_any",
                vec![list(
                    vec![
                        read_global(HOOK_FN, any()),
                        box_i64(read_global(HOOK_MASK, i64())),
                        box_i64(read_global(HOOK_COUNT_EVERY, i64())),
                        box_i64(read_global(HOOK_LEFT, i64())),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    // The hook `x` packed becomes the running thread's.
    let hook_field = |i: i64| {
        at(
            call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
            int(i),
        )
    };
    d.push(define(
        "zl_dbg_hook_use",
        &[x],
        unit(),
        vec![
            if_(
                is_nil(x.e()),
                vec![
                    set_global(HOOK_FN, nil()),
                    set_global(HOOK_MASK, int(0)),
                    set_global(HOOK_COUNT_EVERY, int(0)),
                    set_global(HOOK_LEFT, int(0)),
                ],
                vec![
                    set_global(HOOK_FN, hook_field(0)),
                    set_global(
                        HOOK_MASK,
                        call("zb_box_get_i64", vec![hook_field(1)], i64()),
                    ),
                    set_global(
                        HOOK_COUNT_EVERY,
                        call("zb_box_get_i64", vec![hook_field(2)], i64()),
                    ),
                    set_global(
                        HOOK_LEFT,
                        call("zb_box_get_i64", vec![hook_field(3)], i64()),
                    ),
                ],
            ),
            ret_void(),
        ],
    ));
    // The running thread's hook is kept as thread `k`'s.
    d.push(define("zl_dbg_hook_save", &[k], unit(), {
        let mut st = table_of(HOOKS);
        st.extend([
            expr(call(
                "zl_rawseti",
                vec![
                    unbox_table(read_global(HOOKS, any()), t),
                    k.e(),
                    call("zl_dbg_hook_pack", vec![], any()),
                ],
                unit(),
            )),
            ret_void(),
        ]);
        st
    }));
    // Thread `k`'s hook becomes the running one.
    d.push(define(
        "zl_dbg_hook_load",
        &[k],
        unit(),
        vec![
            x.decl(nil()),
            when(
                not(is_nil(read_global(HOOKS, any()))),
                vec![x.set(call(
                    "zl_rawgeti",
                    vec![unbox_table(read_global(HOOKS, any()), t), k.e()],
                    any(),
                ))],
            ),
            expr(call("zl_dbg_hook_use", vec![x.e()], unit())),
            ret_void(),
        ],
    ));
    // A thread is made with handle `k`: what a dead thread with that
    // handle left goes, and the new one has the running thread's hook
    // mask and count, without its function.
    d.push(define(
        "zl_dbg_new_thread",
        &[k],
        unit(),
        vec![
            when(
                read_global(DBG_ON, boolean()),
                vec![expr(call("zl_dbg_new_thread_raw", vec![k.e()], unit()))],
            ),
            when(
                ne(read_global(HOOK_MASK, i64()), int(0)),
                vec![
                    x.decl(read_global(HOOK_FN, any())),
                    set_global(HOOK_FN, nil()),
                    expr(call("zl_dbg_hook_save", vec![k.e()], unit())),
                    set_global(HOOK_FN, x.e()),
                ],
            ),
            ret_void(),
        ],
    ));
    // The running thread becomes the one with handle `k`, with its
    // stack and its hook.
    d.push(define(
        "zl_dbg_switch",
        &[k],
        unit(),
        vec![
            when(
                read_global(DBG_ON, boolean()),
                vec![
                    set_global(DBG_SITE, int(0)),
                    expr(call("zl_dbg_switch_raw", vec![k.e(), line_now()], unit())),
                    when(
                        or(
                            not(is_nil(read_global(HOOKS, any()))),
                            ne(read_global(HOOK_MASK, i64()), int(0)),
                        ),
                        vec![
                            expr(call(
                                "zl_dbg_hook_save",
                                vec![read_global(HOOK_THREAD, i64())],
                                unit(),
                            )),
                            expr(call("zl_dbg_hook_load", vec![k.e()], unit())),
                        ],
                    ),
                    set_global(HOOK_THREAD, k.e()),
                ],
            ),
            ret_void(),
        ],
    ));
    // Thread `k` is done, killed by an error when `failed`: a hook it
    // had moves to its record `r`, whose handle may be another
    // thread's from now on.
    let r = kept("r", anys.clone());
    let failed = local("failed", boolean());
    d.push(define(
        "zl_dbg_drop",
        &[k, &r, &failed],
        unit(),
        vec![
            when(
                read_global(DBG_ON, boolean()),
                vec![expr(call(
                    "zl_dbg_drop_raw",
                    vec![k.e(), failed.e()],
                    unit(),
                ))],
            ),
            when(
                not(is_nil(read_global(HOOKS, any()))),
                vec![
                    x.decl(call(
                        "zl_rawgeti",
                        vec![unbox_table(read_global(HOOKS, any()), t), k.e()],
                        any(),
                    )),
                    when(
                        not(is_nil(x.e())),
                        vec![
                            expr(call(
                                "zb_list_extend_any",
                                vec![r.e(), list(vec![x.e()], anys.clone())],
                                unit(),
                            )),
                            expr(call(
                                "zl_rawseti",
                                vec![unbox_table(read_global(HOOKS, any()), t), k.e(), nil()],
                                unit(),
                            )),
                        ],
                    ),
                ],
            ),
            ret_void(),
        ],
    ));
}
