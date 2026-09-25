//! The `debug` library over the call stack a program that uses it
//! keeps.
//!
//! Such a program is lowered to record every call: each function
//! pushes a frame on entry (its key, the caller's line, the call
//! site) and pops it on every way out, restoring the caller's line;
//! each call site stores its number first. The host keeps the frames
//! and what the lowering says of each chunk's functions and sites, and
//! answers `traceback` and `getinfo` from them. Upvalues are read and
//! written through an accessor each chunk hands over, which knows
//! where each function keeps each of them; the locals of a frame
//! through the list it spilled them to at the call it stopped at.
//! Hooks are called from the frame pushes and pops and from each
//! statement's line store.
//!
//! A program that never reaches the library pushes nothing; the one
//! cost it pays is a test of `zl_dbg_on` around a coroutine switch.

use super::*;

/// The call site the next call is made from: its chunk above the low
/// 32 bits, its number below, `TAIL_SITE` set for a tail call; 0 for
/// a call the library makes.
pub const DBG_SITE: &str = "zl_dbg_site";
/// Whether the program keeps the call stack.
pub const DBG_ON: &str = "zl_dbg_on";
/// The events a hook is set for: `HOOK_CALL`, `HOOK_RETURN`,
/// `HOOK_LINE`, `HOOK_COUNT`; 0 for no hook.
pub const HOOK_MASK: &str = "zl_hook_mask";
pub const HOOK_CALL: i64 = 1;
pub const HOOK_RETURN: i64 = 2;
pub const HOOK_LINE: i64 = 4;
pub const HOOK_COUNT: i64 = 8;
/// The bit of a site marking a tail call.
pub const TAIL_SITE: i64 = 1 << 31;
/// The site a hook is called from.
pub const HOOK_SITE: i64 = -2;
/// What an upvalue accessor is asked: the value, a store, the
/// upvalue's identity, a join to another's cell, the cell itself.
pub const UP_GET: i64 = 0;
pub const UP_SET: i64 = 1;
pub const UP_ID: i64 = 2;
pub const UP_JOIN: i64 = 3;
pub const UP_CELL: i64 = 4;
/// How many arguments an upvalue accessor takes: the key, the
/// function, the upvalue's number, a value and what is asked.
#[allow(dead_code)]
pub const UP_ARITY: i64 = 5;

const HOOK_FN: &str = "zl_hook_fn";
const HOOK_COUNT_EVERY: &str = "zl_hook_count";
const HOOK_LEFT: &str = "zl_hook_left";
const HOOK_BUSY: &str = "zl_hook_busy";
/// Set by a library function's value wrapper for the `debug` function
/// it calls: its argument errors name it as its library's field,
/// `debug.getinfo`, as the reference names a function it finds no
/// call site's name for.
pub const QUALIFY: &str = "zl_dbg_qualify";
/// Whether `debug.setmetatable` ever gave a type other than tables and
/// strings a metatable: until then no such value has one to consult.
pub const TYPE_METAS_ON: &str = "zl_type_metas_on";
/// The hooks of the threads not running, by handle: each a list of the
/// hook, its mask, its count and what is left of the count. The
/// `zl_hook_*` globals hold the running thread's.
const HOOKS: &str = "zl_hooks";
/// The running thread's handle, 0 for the main one.
const HOOK_THREAD: &str = "zl_hook_thread";
/// The last function value made for each of the program's functions,
/// by key: what `getinfo(level, "f")` answers.
const FUNCS: &str = "zl_dbg_funcs";
/// Each chunk's upvalue accessor, by the chunk's number.
const UPS: &str = "zl_dbg_ups";
/// The globals holding the metatables of nil, booleans, numbers,
/// functions and threads, in that order.
const TYPE_METAS: [(&str, &str); 5] = [
    ("nil", "zl_meta_nil"),
    ("boolean", "zl_meta_boolean"),
    ("number", "zl_meta_number"),
    ("function", "zl_meta_function"),
    ("thread", "zl_meta_thread"),
];
pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let table = t.table();
    let x = kept("x", any());
    let y = kept("y", any());
    let z = kept("z", any());
    let f = kept("f", any());
    let args = kept("args", anys.clone());
    let key = local("key", i64());
    let site = local("site", i64());
    let line = local("line", i64());
    let level = local("level", i64());
    let thread = local("thread", i64());
    let n = local("n", i64());
    let i = local("i", i64());
    let j = local("j", i64());
    let k = local("k", i64());
    let kind = local("kind", i64());
    let first = local("first", i64());
    let what = kept("what", string());
    let name = kept("name", string());
    let source = kept("source", string());
    let short = kept("short", string());
    let meta = kept("meta", string());
    let event = kept("event", string());
    let tb = kept("t", table.clone());
    let lines = kept("lines", table.clone());
    let acc = kept("acc", any());
    let mut d = vec![
        global_var(DBG_SITE, i64()),
        global_var(DBG_ON, boolean()),
        global_var(HOOK_MASK, i64()),
        global_var(HOOK_FN, any()),
        global_var(HOOK_COUNT_EVERY, i64()),
        global_var(HOOK_LEFT, i64()),
        global_var(HOOK_BUSY, boolean()),
        global_var(HOOKS, any()),
        global_var(QUALIFY, boolean()),
        global_var(TYPE_METAS_ON, boolean()),
        global_var(HOOK_THREAD, i64()),
        global_var(FUNCS, any()),
        global_var(UPS, any()),
    ];
    let host = |name: &str, params: &[(&str, Type)], ret: Type, symbol: &str| {
        extern_fn(name, params, ret, Some(symbol))
    };
    d.extend([
        host(
            "zl_dbg_chunk_raw",
            &[
                ("index", i64()),
                ("source", string()),
                ("short", string()),
                ("meta", string()),
            ],
            unit(),
            "$Lua$dbg_chunk",
        ),
        host(
            "zl_dbg_push",
            &[("key", i64()), ("line", i64()), ("site", i64())],
            unit(),
            "$Lua$dbg_enter",
        ),
        host("zl_dbg_pop", &[], i64(), "$Lua$dbg_leave"),
        host(
            "zl_dbg_switch_raw",
            &[("to", i64()), ("line", i64())],
            unit(),
            "$Lua$dbg_switch",
        ),
        host(
            "zl_dbg_drop_raw",
            &[("h", i64()), ("failed", boolean())],
            unit(),
            "$Lua$dbg_drop",
        ),
        host(
            "zl_dbg_new_thread_raw",
            &[("h", i64())],
            unit(),
            "$Lua$dbg_new_thread",
        ),
        host("zl_dbg_handler_enter", &[], i64(), "$Lua$dbg_handler_enter"),
        host(
            "zl_dbg_handler_leave",
            &[("depth", i64())],
            unit(),
            "$Lua$dbg_handler_leave",
        ),
        host("zl_dbg_raise_line", &[], i64(), "$Lua$dbg_raise_line"),
        host(
            "zl_dbg_mm_site",
            &[("event", string())],
            i64(),
            "$Lua$dbg_mm_site",
        ),
        host(
            "zl_dbg_line_raw",
            &[("line", i64()), ("back", boolean())],
            boolean(),
            "$Lua$dbg_line",
        ),
        host("zl_dbg_back_raw", &[], unit(), "$Lua$dbg_back"),
        host(
            "zl_dbg_note_raise",
            &[("line", i64())],
            unit(),
            "$Lua$dbg_note_raise",
        ),
        host(
            "zl_dbg_set_line_raw",
            &[("line", i64())],
            unit(),
            "$Lua$dbg_set_line",
        ),
        host(
            "zl_dbg_traceback_raw",
            &[
                ("message", string()),
                ("has", boolean()),
                ("level", i64()),
                ("thread", i64()),
                ("line", i64()),
            ],
            string(),
            "$Lua$dbg_traceback",
        ),
        host(
            "zl_dbg_check_options",
            &[("what", string())],
            i64(),
            "$Lua$dbg_check_options",
        ),
        host(
            "zl_dbg_info_level",
            &[
                ("level", i64()),
                ("thread", i64()),
                ("line", i64()),
                ("what", string()),
            ],
            i64(),
            "$Lua$dbg_info_level",
        ),
        host(
            "zl_dbg_info_func",
            &[("key", i64()), ("what", string())],
            unit(),
            "$Lua$dbg_info_func",
        ),
        host("zl_dbg_info_count", &[], i64(), "$Lua$dbg_info_count"),
        host(
            "zl_dbg_info_name",
            &[("i", i64())],
            string(),
            "$Lua$dbg_info_name",
        ),
        host(
            "zl_dbg_info_kind",
            &[("i", i64())],
            i64(),
            "$Lua$dbg_info_kind",
        ),
        host(
            "zl_dbg_info_int",
            &[("i", i64())],
            i64(),
            "$Lua$dbg_info_int",
        ),
        host(
            "zl_dbg_info_str",
            &[("i", i64())],
            string(),
            "$Lua$dbg_info_str",
        ),
        host(
            "zl_dbg_info_line",
            &[("i", i64()), ("j", i64())],
            i64(),
            "$Lua$dbg_info_line",
        ),
        host("zl_dbg_info_key", &[], i64(), "$Lua$dbg_info_key"),
        host(
            "zl_dbg_upvalue_name",
            &[("key", i64()), ("n", i64())],
            string(),
            "$Lua$dbg_upvalue_name",
        ),
        host(
            "zl_dbg_frame_depth",
            &[("level", i64())],
            i64(),
            "$Lua$dbg_frame_depth",
        ),
        host(
            "zl_dbg_spill",
            &[("list", any()), ("site", i64())],
            unit(),
            "$Lua$dbg_spill",
        ),
        host(
            "zl_dbg_unspill",
            &[("list", any())],
            i64(),
            "$Lua$dbg_unspill",
        ),
        host(
            "zl_dbg_mark_set",
            &[("depth", i64()), ("i", i64())],
            unit(),
            "$Lua$dbg_mark_set",
        ),
        host(
            "zl_dbg_frame_spill",
            &[("depth", i64())],
            i64(),
            "$Lua$dbg_frame_spill",
        ),
        host(
            "zl_dbg_local_name",
            &[("depth", i64()), ("n", i64())],
            string(),
            "$Lua$dbg_local_name",
        ),
        host(
            "zl_dbg_local_count",
            &[("depth", i64())],
            i64(),
            "$Lua$dbg_local_count",
        ),
        host(
            "zl_dbg_local_cell",
            &[("depth", i64()), ("n", i64())],
            i64(),
            "$Lua$dbg_local_cell",
        ),
        host(
            "zl_dbg_param_name",
            &[("key", i64()), ("n", i64())],
            string(),
            "$Lua$dbg_param_name",
        ),
        host(
            "zl_dbg_frame_key",
            &[("depth", i64())],
            i64(),
            "$Lua$dbg_frame_key",
        ),
        host("zl_dbg_word_as_any", &[("w", i64())], any(), "$Lua$word"),
    ]);

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
    let thread_handle = |co: Expr| {
        call(
            "zb_box_get_i64",
            vec![at(
                call("zb_unbox_list_raw_any", vec![co], anys.clone()),
                int(0),
            )],
            i64(),
        )
    };

    // ─── metatables of the other types ──────────────────────────
    // The metatable `debug.setmetatable` gave a type that is neither
    // a table nor a string: nil when none.
    for (_, global) in TYPE_METAS {
        d.push(global_var(global, any()));
    }
    let type_meta = |x: Expr| -> Vec<Stmt> {
        vec![
            when(
                is_nil(x.clone()),
                vec![ret(read_global(TYPE_METAS[0].1, any()))],
            ),
            when(
                eq(category(x.clone()), int(BOOL)),
                vec![ret(read_global(TYPE_METAS[1].1, any()))],
            ),
            when(
                or(
                    eq(category(x.clone()), int(INT)),
                    or(
                        eq(category(x.clone()), int(UINT)),
                        eq(category(x.clone()), int(FLOAT)),
                    ),
                ),
                vec![ret(read_global(TYPE_METAS[2].1, any()))],
            ),
            when(
                is_func(x.clone()),
                vec![ret(read_global(TYPE_METAS[3].1, any()))],
            ),
            when(is_thread(x), vec![ret(read_global(TYPE_METAS[4].1, any()))]),
            ret(nil()),
        ]
    };
    d.push(define("zl_type_meta", &[&x], any(), type_meta(x.e())));
    // Gives `x`'s type the metatable `y`; whether `x` has such a type.
    d.push(define(
        "zl_set_type_meta",
        &[&x, &y],
        boolean(),
        vec![
            set_global(TYPE_METAS_ON, bool(true)),
            when(
                is_nil(x.e()),
                vec![set_global(TYPE_METAS[0].1, y.e()), ret(bool(true))],
            ),
            when(
                eq(category(x.e()), int(BOOL)),
                vec![set_global(TYPE_METAS[1].1, y.e()), ret(bool(true))],
            ),
            when(
                or(
                    eq(category(x.e()), int(INT)),
                    or(
                        eq(category(x.e()), int(UINT)),
                        eq(category(x.e()), int(FLOAT)),
                    ),
                ),
                vec![set_global(TYPE_METAS[2].1, y.e()), ret(bool(true))],
            ),
            when(
                is_func(x.e()),
                vec![set_global(TYPE_METAS[3].1, y.e()), ret(bool(true))],
            ),
            when(
                is_thread(x.e()),
                vec![set_global(TYPE_METAS[4].1, y.e()), ret(bool(true))],
            ),
            ret(bool(false)),
        ],
    ));

    // ─── the stack ──────────────────────────────────────────────
    // A chunk's functions and call sites, and the accessor of its
    // functions' upvalues.
    d.push(define(
        "zl_dbg_register",
        &[&k, &source, &short, &meta, &acc],
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
    d.push(define("zl_dbg_closure", &[&key, &f], unit(), {
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
        &[&event, &x],
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
        &[&key],
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
        &[&site],
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
        &[&line],
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
                    vec![
                        text("line"),
                        box_i64(bitand(line.e(), int((1i64 << LINE_BITS) - 1))),
                    ],
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
                vec![
                    text("line"),
                    box_i64(bitand(line, int((1i64 << LINE_BITS) - 1))),
                ],
                unit(),
            ))],
        )
    };
    d.push(define(
        "zl_dbg_loop",
        &[&line],
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
        &[&line],
        unit(),
        vec![line_event(line.e(), true), ret_void()],
    ));
    // A loop whose head is on `line` is left: the next statement is a
    // line event only on a line of its own.
    d.push(define(
        "zl_dbg_loop_exit",
        &[&line],
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
        &[&x, &n, &i, &y],
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
        &[&x],
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
    d.push(define("zl_dbg_hook_save", &[&k], unit(), {
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
        &[&k],
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
        &[&k],
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
        &[&k],
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
        &[&k, &r, &failed],
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

    // ─── the library ────────────────────────────────────────────
    // The thread a debug function is asked about, when its first
    // argument is one: its handle and 1 (the arguments after it
    // start one later); else -1 and 0.
    let thread_arg = |args: &Local, thread: &Local, first: &Local| {
        vec![
            thread.decl(int(-1)),
            first.decl(int(0)),
            x.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            when(
                is_thread(x.e()),
                vec![thread.set(thread_handle(x.e())), first.set(int(1))],
            ),
        ]
    };
    let arg = |i: Expr| call("zl_value_at", vec![args.e(), add(first.e(), i)], any());

    // What an argument error calls the function `what`: `debug.what`
    // when it was called as a value.
    let qual = local("qual", boolean());
    let fname = |what: &str| if_expr(qual.e(), text(&format!("debug.{what}")), text(what));
    let bad = |n: usize, what: &str, rest: &str| {
        concat(vec![
            text(&format!("bad argument #{n} to '")),
            fname(what),
            text(&format!("'{rest}")),
        ])
    };
    let qualify = || {
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
        ]
    };

    // `debug.traceback([thread,] [message [, level]])`.
    let mut st = qualify();
    st.extend(thread_arg(&args, &thread, &first));
    st.extend([
        y.decl(arg(int(1))),
        when(
            and(
                not(is_nil(y.e())),
                and(
                    ne(category(y.e()), int(STR)),
                    and(
                        ne(category(y.e()), int(INT)),
                        and(
                            ne(category(y.e()), int(UINT)),
                            ne(category(y.e()), int(FLOAT)),
                        ),
                    ),
                ),
            ),
            vec![ret(y.e())],
        ),
        level.decl(if_expr(eq(thread.e(), int(-1)), int(1), int(0))),
        z.decl(arg(int(2))),
        when(
            not(is_nil(z.e())),
            vec![level.set(call(
                "zl_arg_int",
                vec![z.e(), bad(2, "traceback", "")],
                i64(),
            ))],
        ),
        name.decl(text("")),
        when(
            not(is_nil(y.e())),
            vec![name.set(call("zl_tostring", vec![y.e()], string()))],
        ),
        ret(box_str(call(
            "zl_dbg_traceback_raw",
            vec![
                name.e(),
                not(is_nil(y.e())),
                level.e(),
                thread.e(),
                line_now(),
            ],
            string(),
        ))),
    ]);
    d.push(define("zl_debug_traceback", &[&args], any(), st));

    // `debug.getinfo([thread,] f | level [, what])`.
    let mut st = qualify();
    st.extend(thread_arg(&args, &thread, &first));
    st.extend([
        y.decl(arg(int(1))),
        z.decl(arg(int(2))),
        what.decl(text("flnSrtu")),
        when(
            not(is_nil(z.e())),
            vec![
                when(
                    and(
                        ne(category(z.e()), int(STR)),
                        is_nil(call("zl_arith_operand", vec![z.e()], any())),
                    ),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                        text(" to '"),
                        fname("getinfo"),
                        text("' (string expected, got "),
                        arg_type_name(z.e()),
                        text(")"),
                    ]))],
                ),
                what.set(call("zl_tostring", vec![z.e()], string())),
            ],
        ),
        kind.decl(call("zl_dbg_check_options", vec![what.e()], i64())),
        when(
            ne(kind.e(), int(0)),
            vec![lua_error(concat(vec![
                text("bad argument #"),
                call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                text(" to '"),
                fname("getinfo"),
                text("' (invalid option"),
                if_expr(eq(kind.e(), int(2)), text(" '>')"), text(")")),
            ]))],
        ),
        if_(
            is_func(y.e()),
            vec![expr(call(
                "zl_dbg_info_func",
                vec![call("zl_func_id", vec![y.e()], i64()), what.e()],
                unit(),
            ))],
            vec![
                when(
                    lt(len(args.e()), add(first.e(), int(1))),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(1))], string()),
                        text(" to '"),
                        fname("getinfo"),
                        text("' (number expected, got no value)"),
                    ]))],
                ),
                level.decl(call(
                    "zl_arg_int",
                    vec![
                        y.e(),
                        concat(vec![
                            text("bad argument #"),
                            call("zb_str_of_int", vec![add(first.e(), int(1))], string()),
                            text(" to '"),
                            fname("getinfo"),
                            text("'"),
                        ]),
                    ],
                    i64(),
                )),
                when(
                    eq(
                        call(
                            "zl_dbg_info_level",
                            vec![level.e(), thread.e(), line_now(), what.e()],
                            i64(),
                        ),
                        int(0),
                    ),
                    vec![ret(nil())],
                ),
            ],
        ),
        tb.decl(call("zl_table_new", vec![], table.clone())),
        n.decl(call("zl_dbg_info_count", vec![], i64())),
        i.decl(int(0)),
        while_(
            lt(i.e(), n.e()),
            vec![
                name.decl(call("zl_dbg_info_name", vec![i.e()], string())),
                kind.set(call("zl_dbg_info_kind", vec![i.e()], i64())),
                x.set(nil()),
                when(
                    eq(kind.e(), int(1)),
                    vec![x.set(box_i64(call("zl_dbg_info_int", vec![i.e()], i64())))],
                ),
                when(
                    eq(kind.e(), int(2)),
                    vec![x.set(box_str(call("zl_dbg_info_str", vec![i.e()], string())))],
                ),
                when(
                    eq(kind.e(), int(3)),
                    vec![x.set(box_bool(ne(
                        call("zl_dbg_info_int", vec![i.e()], i64()),
                        int(0),
                    )))],
                ),
                when(
                    eq(kind.e(), int(4)),
                    vec![if_(
                        is_func(y.e()),
                        vec![x.set(y.e())],
                        vec![
                            key.decl(call("zl_dbg_info_key", vec![], i64())),
                            when(
                                and(ne(key.e(), int(-1)), not(is_nil(read_global(FUNCS, any())))),
                                vec![x.set(call(
                                    "zl_rawgeti",
                                    vec![unbox_table(read_global(FUNCS, any()), t), key.e()],
                                    any(),
                                ))],
                            ),
                        ],
                    )],
                ),
                when(
                    eq(kind.e(), int(5)),
                    vec![
                        lines.decl(call("zl_table_new", vec![], table.clone())),
                        k.decl(call("zl_dbg_info_int", vec![i.e()], i64())),
                        j.decl(int(0)),
                        while_(
                            lt(j.e(), k.e()),
                            vec![
                                expr(call(
                                    "zl_rawseti",
                                    vec![
                                        lines.e(),
                                        call("zl_dbg_info_line", vec![i.e(), j.e()], i64()),
                                        box_bool(bool(true)),
                                    ],
                                    unit(),
                                )),
                                j.add_assign(int(1)),
                            ],
                        ),
                        x.set(box_table(lines.e())),
                    ],
                ),
                when(
                    not(is_nil(x.e())),
                    vec![expr(call(
                        "zl_rawset_str",
                        vec![tb.e(), name.e(), x.e()],
                        unit(),
                    ))],
                ),
                i.add_assign(int(1)),
            ],
        ),
        ret(box_table(tb.e())),
    ]);
    d.push(define("zl_debug_getinfo", &[&args], any(), st));

    // ─── upvalues ───────────────────────────────────────────────
    // The accessor of the chunk a function of the program belongs to.
    d.push(define(
        "zl_dbg_accessor",
        &[&key],
        any(),
        vec![
            when(
                or(lt(key.e(), int(0)), is_nil(read_global(UPS, any()))),
                vec![ret(nil())],
            ),
            ret(call(
                "zl_rawgeti",
                vec![
                    unbox_table(read_global(UPS, any()), t),
                    shr(key.e(), int(32)),
                ],
                any(),
            )),
        ],
    ));
    // Upvalue `n` of `f` asked `how`: what the chunk's accessor answers.
    let how = local("how", i64());
    d.push(define(
        "zl_dbg_upvalue",
        &[&f, &n, &x, &how],
        any(),
        vec![
            key.decl(call("zl_func_id", vec![f.e()], i64())),
            acc.decl(call("zl_dbg_accessor", vec![key.e()], any())),
            when(is_nil(acc.e()), vec![ret(nil())]),
            ret(call(
                "zl_apply_5",
                vec![
                    acc.e(),
                    box_i64(key.e()),
                    f.e(),
                    box_i64(n.e()),
                    x.e(),
                    box_i64(how.e()),
                ],
                any(),
            )),
        ],
    ));
    // The name of upvalue `n` of `f`: empty when it has none.
    d.push(define(
        "zl_dbg_upvalue_name_of",
        &[&f, &n],
        string(),
        vec![
            key.decl(call("zl_func_id", vec![f.e()], i64())),
            when(lt(key.e(), int(0)), vec![ret(text(""))]),
            ret(call("zl_dbg_upvalue_name", vec![key.e(), n.e()], string())),
        ],
    ));
    let function_arg = |x: &Local, what: &str, number: usize| {
        when(
            not(is_func(x.e())),
            vec![lua_error(concat(vec![
                bad(number, what, " (function expected, got "),
                arg_type_name(x.e()),
                text(")"),
            ]))],
        )
    };
    let int_arg = |x: Expr, what: &str, number: usize| {
        call("zl_arg_int", vec![x, bad(number, what, "")], i64())
    };
    let none = || call("zl_none", vec![], any());
    let pair = |a: Expr, b: Expr| call("zb_box_tuple", vec![list(vec![a, b], anys.clone())], any());
    // `debug.getupvalue(f, n)`: the name and value, or nothing.
    d.push(define(
        "zl_debug_getupvalue",
        &[&args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(&f, "getupvalue", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "getupvalue",
                2,
            )),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(none())],
            ),
            ret(pair(
                box_str(name.e()),
                call(
                    "zl_dbg_upvalue",
                    vec![f.e(), n.e(), nil(), int(UP_GET)],
                    any(),
                ),
            )),
        ],
    ));
    // `debug.setupvalue(f, n, v)`: the name, or nothing.
    d.push(define(
        "zl_debug_setupvalue",
        &[&args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(&f, "setupvalue", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "setupvalue",
                2,
            )),
            when(
                lt(len(args.e()), int(3)),
                vec![lua_error(bad(3, "setupvalue", " (value expected)"))],
            ),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(none())],
            ),
            expr(call(
                "zl_dbg_upvalue",
                vec![
                    f.e(),
                    n.e(),
                    call("zl_value_at", vec![args.e(), int(3)], any()),
                    int(UP_SET),
                ],
                any(),
            )),
            ret(box_str(name.e())),
        ],
    ));
    // `debug.upvalueid(f, n)`: what tells upvalues apart.
    d.push(define(
        "zl_debug_upvalueid",
        &[&args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(&f, "upvalueid", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "upvalueid",
                2,
            )),
            name.decl(call("zl_dbg_upvalue_name_of", vec![f.e(), n.e()], string())),
            when(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                vec![ret(call("zl_fail", vec![], any()))],
            ),
            ret(call(
                "zl_dbg_upvalue",
                vec![f.e(), n.e(), nil(), int(UP_ID)],
                any(),
            )),
        ],
    ));
    // `debug.upvaluejoin(f1, n1, f2, n2)`: upvalue `n1` of `f1` is
    // upvalue `n2` of `f2` from now on.
    let check_index = |f: Expr, n: Expr, number: usize| {
        when(
            eq(
                call(
                    "zb_str_len",
                    vec![call("zl_dbg_upvalue_name_of", vec![f, n], string())],
                    i64(),
                ),
                int(0),
            ),
            vec![lua_error(bad(
                number,
                "upvaluejoin",
                " (invalid upvalue index)",
            ))],
        )
    };
    let lua_function = |f: Expr, number: usize| {
        when(
            lt(call("zl_func_id", vec![f], i64()), int(0)),
            vec![lua_error(bad(
                number,
                "upvaluejoin",
                " (Lua function expected)",
            ))],
        )
    };
    let g = kept("g", any());
    let m = local("m", i64());
    d.push(define(
        "zl_debug_upvaluejoin",
        &[&args],
        unit(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(&f, "upvaluejoin", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "upvaluejoin",
                2,
            )),
            g.decl(call("zl_value_at", vec![args.e(), int(3)], any())),
            function_arg(&g, "upvaluejoin", 3),
            m.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(4)], any()),
                "upvaluejoin",
                4,
            )),
            check_index(f.e(), n.e(), 2),
            check_index(g.e(), m.e(), 4),
            lua_function(f.e(), 1),
            lua_function(g.e(), 3),
            x.decl(call(
                "zl_dbg_upvalue",
                vec![g.e(), m.e(), nil(), int(UP_CELL)],
                any(),
            )),
            expr(call(
                "zl_dbg_upvalue",
                vec![f.e(), n.e(), x.e(), int(UP_JOIN)],
                any(),
            )),
            ret_void(),
        ],
    ));

    // ─── locals ─────────────────────────────────────────────────
    // Local `n` of stack level `level`: where its value lives in the
    // list the frame spilled its locals to (a list of the values, the
    // extra arguments of a variadic function boxed after them), its
    // name, and the list; an empty name when there is no such local.
    // `n` below zero counts the extra arguments.
    let depth = local("depth", i64());
    let spill = borrowed("spill", anys.clone());
    let lookup = |who: &str, st: &mut Vec<Stmt>| {
        st.extend([
            depth.decl(call("zl_dbg_frame_depth", vec![level.e()], i64())),
            when(
                lt(depth.e(), int(0)),
                vec![lua_error(bad(1, who, " (level out of range)"))],
            ),
            name.decl(text("")),
            x.decl(nil()),
            k.decl(int(0)),
            when(
                gt(depth.e(), int(0)),
                vec![
                    k.set(call("zl_dbg_frame_spill", vec![depth.e()], i64())),
                    when(
                        ne(k.e(), int(0)),
                        vec![x.set(call("zl_dbg_word_as_any", vec![k.e()], any()))],
                    ),
                ],
            ),
        ]);
    };
    // Reads `name` and the slot `i` of `spill` the local is at, or
    // leaves `name` empty.
    let find = |st: &mut Vec<Stmt>| {
        st.push(when(
            not(is_nil(x.e())),
            vec![
                spill.decl(call("zb_unbox_list_raw_any", vec![x.e()], anys.clone())),
                j.decl(call("zl_dbg_local_count", vec![depth.e()], i64())),
                if_(
                    gt(n.e(), int(0)),
                    vec![when(
                        and(le(n.e(), j.e()), le(n.e(), len(spill.e()))),
                        vec![
                            name.set(call("zl_dbg_local_name", vec![depth.e(), n.e()], string())),
                            i.set(sub(n.e(), int(1))),
                        ],
                    )],
                    vec![when(
                        and(lt(n.e(), int(0)), gt(len(spill.e()), j.e())),
                        vec![
                            y.set(at(spill.e(), j.e())),
                            z.set(call(
                                "zl_value_at",
                                vec![call("zl_values", vec![y.e()], anys.clone()), neg(n.e())],
                                any(),
                            )),
                            when(
                                le(
                                    neg(n.e()),
                                    len(call("zl_values", vec![y.e()], anys.clone())),
                                ),
                                vec![name.set(text("(vararg)")), i.set(int(-1))],
                            ),
                        ],
                    )],
                ),
            ],
        ));
    };
    // `debug.getlocal([thread,] level | f, n)`.
    let mut st = qualify();
    st.extend(thread_arg(&args, &thread, &first));
    st.extend([
        y.decl(arg(int(1))),
        z.decl(nil()),
        when(
            lt(len(args.e()), add(first.e(), int(2))),
            vec![lua_error(bad(
                2,
                "getlocal",
                " (number expected, got no value)",
            ))],
        ),
        n.decl(int_arg(arg(int(2)), "getlocal", 2)),
        i.decl(int(0)),
        j.decl(int(0)),
        // A function: the name of its parameter `n`.
        when(
            is_func(y.e()),
            vec![
                name.decl(call(
                    "zl_dbg_param_name",
                    vec![call("zl_func_id", vec![y.e()], i64()), n.e()],
                    string(),
                )),
                when(
                    eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                    vec![ret(call("zl_fail", vec![], any()))],
                ),
                ret(box_str(name.e())),
            ],
        ),
        // Another thread's frames keep no locals here.
        when(
            ne(thread.e(), int(-1)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        level.decl(call(
            "zl_arg_int",
            vec![y.e(), bad(1, "getlocal", "")],
            i64(),
        )),
    ]);
    lookup("getlocal", &mut st);
    find(&mut st);
    st.extend([
        when(
            eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        when(
            eq(i.e(), int(-1)),
            vec![ret(pair(box_str(name.e()), z.e()))],
        ),
        y.set(at(
            call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
            i.e(),
        )),
        // A local kept in a cell was spilled as the cell.
        when(
            ne(
                call("zl_dbg_local_cell", vec![depth.e(), n.e()], i64()),
                int(0),
            ),
            vec![y.set(at(
                call("zb_unbox_list_raw_any", vec![y.e()], anys.clone()),
                int(0),
            ))],
        ),
        ret(pair(box_str(name.e()), y.e())),
    ]);
    d.push(define("zl_debug_getlocal", &[&args], any(), st));
    // `debug.setlocal([thread,] level, n, v)`: the name, or fail.
    let mut st = qualify();
    st.extend(thread_arg(&args, &thread, &first));
    st.extend([
        y.decl(nil()),
        z.decl(nil()),
        level.decl(int_arg(arg(int(1)), "setlocal", 1)),
        n.decl(int_arg(arg(int(2)), "setlocal", 2)),
        when(
            lt(len(args.e()), add(first.e(), int(3))),
            vec![lua_error(bad(3, "setlocal", " (value expected)"))],
        ),
        when(
            ne(thread.e(), int(-1)),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        i.decl(int(0)),
        j.decl(int(0)),
    ]);
    lookup("setlocal", &mut st);
    find(&mut st);
    st.extend([
        when(
            or(
                eq(call("zb_str_len", vec![name.e()], i64()), int(0)),
                lt(i.e(), int(0)),
            ),
            vec![ret(call("zl_fail", vec![], any()))],
        ),
        // A local kept in a cell is set through it, where every closure
        // sharing it sees the value; any other is read back by its
        // frame when the call it stopped at returns.
        if_(
            ne(
                call("zl_dbg_local_cell", vec![depth.e(), n.e()], i64()),
                int(0),
            ),
            vec![set_idx(
                call(
                    "zb_unbox_list_raw_any",
                    vec![at(
                        call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                        i.e(),
                    )],
                    anys.clone(),
                ),
                int(0),
                arg(int(3)),
            )],
            vec![
                set_idx(
                    call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                    i.e(),
                    arg(int(3)),
                ),
                expr(call("zl_dbg_mark_set", vec![depth.e(), i.e()], unit())),
            ],
        ),
        ret(box_str(name.e())),
    ]);
    d.push(define("zl_debug_setlocal", &[&args], any(), st));

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = qualify();
    st.extend(thread_arg(&args, &thread, &first));
    let other = local("other", boolean());
    // The thread argument `x`'s record, and whether it is dead: its
    // handle may then be another thread's.
    let thread_record = || call("zb_unbox_list_raw_any", vec![x.e()], anys.clone());
    let thread_dead = || {
        eq(
            call(
                "zb_box_get_i64",
                vec![at(thread_record(), int(coroutines::STATUS))],
                i64(),
            ),
            int(coroutines::DEAD),
        )
    };
    // A hook for another thread is set as the running one's, then
    // kept as that thread's (a dead one's in its record) while the
    // running one's comes back.
    let other_tail = || {
        when(
            other.e(),
            vec![
                if_(
                    thread_dead(),
                    vec![if_(
                        gt(len(thread_record()), int(coroutines::DEAD_HOOK)),
                        vec![set_idx(
                            thread_record(),
                            int(coroutines::DEAD_HOOK),
                            call("zl_dbg_hook_pack", vec![], any()),
                        )],
                        vec![expr(call(
                            "zb_list_extend_any",
                            vec![
                                thread_record(),
                                list(vec![call("zl_dbg_hook_pack", vec![], any())], anys.clone()),
                            ],
                            unit(),
                        ))],
                    )],
                    vec![expr(call("zl_dbg_hook_save", vec![thread.e()], unit()))],
                ),
                expr(call(
                    "zl_dbg_hook_load",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
            ],
        )
    };
    let has_char = |s: Expr, c: &str| ge(call("zb_str_index_of", vec![s, text(c)], i64()), int(0));
    // The hook is told of `sethook`'s own call and return, as of any
    // function's: its call under the hook it replaces, its return
    // under the one it sets.
    let own_event = |bit: i64, event: &str| {
        when(
            has_event(bit),
            vec![
                expr(call(
                    "zl_dbg_push",
                    vec![int(-1), line_now(), int(0)],
                    unit(),
                )),
                expr(call("zl_dbg_hook", vec![text(event), nil()], unit())),
                set_global(LINE, call("zl_dbg_pop", vec![], i64())),
            ],
        )
    };
    st.extend([
        own_event(HOOK_CALL, "call"),
        other.decl(and(
            ne(thread.e(), int(-1)),
            ne(thread.e(), read_global(HOOK_THREAD, i64())),
        )),
        when(
            other.e(),
            vec![expr(call(
                "zl_dbg_hook_save",
                vec![read_global(HOOK_THREAD, i64())],
                unit(),
            ))],
        ),
        f.decl(arg(int(1))),
        y.decl(arg(int(2))),
        z.decl(arg(int(3))),
        when(
            or(is_nil(f.e()), is_nil(y.e())),
            vec![
                set_global(HOOK_MASK, int(0)),
                set_global(HOOK_FN, nil()),
                set_global(HOOK_COUNT_EVERY, int(0)),
                other_tail(),
                ret_void(),
            ],
        ),
        function_arg(&f, "sethook", 1),
        name.decl(call(
            "zl_arg_str",
            vec![y.e(), bad(2, "sethook", "")],
            string(),
        )),
        k.decl(int(0)),
        when(
            not(is_nil(z.e())),
            vec![
                k.set(int_arg(z.e(), "sethook", 3)),
                when(not(is_nil(pending())), vec![ret_void()]),
            ],
        ),
        m.decl(int(0)),
        when(
            has_char(name.e(), "c"),
            vec![m.set(bitor(m.e(), int(HOOK_CALL)))],
        ),
        when(
            has_char(name.e(), "r"),
            vec![m.set(bitor(m.e(), int(HOOK_RETURN)))],
        ),
        when(
            has_char(name.e(), "l"),
            vec![m.set(bitor(m.e(), int(HOOK_LINE)))],
        ),
        when(
            gt(k.e(), int(0)),
            vec![m.set(bitor(m.e(), int(HOOK_COUNT)))],
        ),
        set_global(HOOK_FN, f.e()),
        set_global(HOOK_COUNT_EVERY, k.e()),
        set_global(HOOK_LEFT, k.e()),
        set_global(HOOK_MASK, if_expr(eq(m.e(), int(0)), int(0), m.e())),
        when(eq(m.e(), int(0)), vec![set_global(HOOK_FN, nil())]),
        other_tail(),
        own_event(HOOK_RETURN, "return"),
        ret_void(),
    ]);
    d.push(define("zl_debug_sethook", &[&args], unit(), st));
    // `debug.gethook([thread])`: the hook, its mask and its count, or
    // fail when none is set.
    let mut st = thread_arg(&args, &thread, &first);
    st.extend([
        when(
            and(
                ne(thread.e(), int(-1)),
                ne(thread.e(), read_global(HOOK_THREAD, i64())),
            ),
            vec![
                expr(call(
                    "zl_dbg_hook_save",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
                if_(
                    thread_dead(),
                    vec![expr(call(
                        "zl_dbg_hook_use",
                        vec![if_expr(
                            gt(len(thread_record()), int(coroutines::DEAD_HOOK)),
                            at(thread_record(), int(coroutines::DEAD_HOOK)),
                            nil(),
                        )],
                        unit(),
                    ))],
                    vec![expr(call("zl_dbg_hook_load", vec![thread.e()], unit()))],
                ),
                y.decl(call("zl_dbg_gethook_now", vec![], any())),
                expr(call(
                    "zl_dbg_hook_load",
                    vec![read_global(HOOK_THREAD, i64())],
                    unit(),
                )),
                ret(y.e()),
            ],
        ),
        ret(call("zl_dbg_gethook_now", vec![], any())),
    ]);
    d.push(define("zl_debug_gethook", &[&args], any(), st));
    // The running thread's hook, its mask and its count, or fail.
    d.push(define(
        "zl_dbg_gethook_now",
        &[],
        any(),
        vec![
            when(
                and(
                    is_nil(read_global(HOOK_FN, any())),
                    eq(read_global(HOOK_MASK, i64()), int(0)),
                ),
                vec![ret(call("zl_fail", vec![], any()))],
            ),
            name.decl(text("")),
            when(
                has_event(HOOK_CALL),
                vec![name.set(add(name.e(), text("c")))],
            ),
            when(
                has_event(HOOK_RETURN),
                vec![name.set(add(name.e(), text("r")))],
            ),
            when(
                has_event(HOOK_LINE),
                vec![name.set(add(name.e(), text("l")))],
            ),
            ret(call(
                "zb_box_tuple",
                vec![list(
                    vec![
                        read_global(HOOK_FN, any()),
                        box_str(name.e()),
                        box_i64(read_global(HOOK_COUNT_EVERY, i64())),
                    ],
                    anys.clone(),
                )],
                any(),
            )),
        ],
    ));
    // `fail`: nil, as one value.
    d.push(define("zl_fail", &[], any(), vec![ret(nil())]));

    // ─── the rest ───────────────────────────────────────────────
    // A userdata's user values. The optional `n` is checked first.
    // The only full userdata is a foreign object, which has no user
    // values: a get answers fail, a set answers fail once it has its
    // value, and a set on anything else refuses its first argument.
    let opt_n = |number: usize, what: &str| {
        expr(call(
            "zl_arg_opt_int",
            vec![
                call("zl_value_at", vec![args.e(), int(number as i64)], any()),
                int(1),
                bad(number, what, ""),
            ],
            i64(),
        ))
    };
    d.push(define(
        "zl_debug_getuservalue",
        &[&args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            opt_n(2, "getuservalue"),
            ret(call("zl_fail", vec![], any())),
        ],
    ));
    d.push(define(
        "zl_debug_setuservalue",
        &[&args],
        any(),
        vec![
            qual.decl(read_global(QUALIFY, boolean())),
            set_global(QUALIFY, bool(false)),
            opt_n(3, "setuservalue"),
            x.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            when(
                zyntax_builtins::foreign::is_foreign(x.e()),
                vec![
                    when(
                        lt(len(args.e()), int(2)),
                        vec![lua_error(bad(2, "setuservalue", " (value expected)"))],
                    ),
                    ret(call("zl_fail", vec![], any())),
                ],
            ),
            lua_error(concat(vec![
                bad(1, "setuservalue", " (userdata expected, got "),
                if_expr(
                    lt(len(args.e()), int(1)),
                    text("no value"),
                    arg_type_name(x.e()),
                ),
                text(")"),
            ])),
            ret(call("zl_fail", vec![], any())),
        ],
    ));
    // `debug.debug()`: each line of the standard input run as a
    // chunk, until `cont` or the input's end; errors are reported and
    // the loop goes on.
    d.push(define(
        "zl_debug_debug",
        &[],
        unit(),
        vec![
            x.decl(nil()),
            y.decl(nil()),
            while_(
                bool(true),
                vec![
                    x.set(call("zl_io_read", vec![list(vec![], anys.clone())], any())),
                    when(
                        or(
                            is_nil(x.e()),
                            or(
                                ne(category(x.e()), int(STR)),
                                str_eq(get_str(x.e()), text("cont")),
                            ),
                        ),
                        vec![ret_void()],
                    ),
                    y.set(call(
                        "zl_load_raw",
                        vec![
                            get_str(x.e()),
                            text("=(debug command)"),
                            call("zl_globals_value", vec![], any()),
                        ],
                        any(),
                    )),
                    if_(
                        is_nil(y.e()),
                        vec![expr(call(
                            "zb_eprintln",
                            vec![call("zl_load_error", vec![], string())],
                            unit(),
                        ))],
                        vec![
                            expr(call("zl_call_0", vec![y.e()], any())),
                            when(
                                not(is_nil(pending())),
                                vec![expr(call(
                                    "zb_eprintln",
                                    vec![call(
                                        "zl_tostring",
                                        vec![call("zl_take_pending", vec![], any())],
                                        string(),
                                    )],
                                    unit(),
                                ))],
                            ),
                        ],
                    ),
                ],
            ),
            ret_void(),
        ],
    ));
    // The C stack's limit is not the program's to set.
    d.push(define(
        "zl_debug_setcstacklimit",
        &[&args],
        i64(),
        vec![ret(int(0))],
    ));
    d
}
