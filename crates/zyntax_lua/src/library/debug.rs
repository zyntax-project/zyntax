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
/// The kind of an upvalue's identity: an instance whose address is
/// the identity.
pub const UPVALUE_ID_KIND: usize = 5;

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
        host("zl_dbg_drop_raw", &[("h", i64())], unit(), "$Lua$dbg_drop"),
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
        host("zl_dbg_raised", &[], string(), "$Lua$dbg_raised"),
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
    // The running thread becomes the one with handle `k`.
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
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(
        "zl_dbg_drop",
        &[&k],
        unit(),
        vec![
            when(
                read_global(DBG_ON, boolean()),
                vec![expr(call("zl_dbg_drop_raw", vec![k.e()], unit()))],
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

    // `debug.traceback([thread,] [message [, level]])`.
    let mut st = thread_arg(&args, &thread, &first);
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
                vec![z.e(), text("bad argument #2 to 'traceback'")],
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
    let mut st = thread_arg(&args, &thread, &first);
    st.extend([
        y.decl(arg(int(1))),
        z.decl(arg(int(2))),
        what.decl(text("flnSrtu")),
        when(
            not(is_nil(z.e())),
            vec![
                when(
                    ne(category(z.e()), int(STR)),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                        text(" to 'getinfo' (string expected, got "),
                        type_name(z.e()),
                        text(")"),
                    ]))],
                ),
                what.set(get_str(z.e())),
            ],
        ),
        kind.decl(call("zl_dbg_check_options", vec![what.e()], i64())),
        when(
            ne(kind.e(), int(0)),
            vec![lua_error(concat(vec![
                text("bad argument #"),
                call("zb_str_of_int", vec![add(first.e(), int(2))], string()),
                text(" to 'getinfo' (invalid option"),
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
                    is_nil(call("zl_arith_operand", vec![y.e()], any())),
                    vec![lua_error(concat(vec![
                        text("bad argument #"),
                        call("zb_str_of_int", vec![add(first.e(), int(1))], string()),
                        text(" to 'getinfo' (function or level expected)"),
                    ]))],
                ),
                level.decl(call(
                    "zl_arg_int",
                    vec![y.e(), text("bad argument #1 to 'getinfo'")],
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
                text(&format!(
                    "bad argument #{number} to '{what}' (function expected, got "
                )),
                type_name(x.e()),
                text(")"),
            ]))],
        )
    };
    let int_arg = |x: Expr, what: &str, number: usize| {
        call(
            "zl_arg_int",
            vec![x, text(&format!("bad argument #{number} to '{what}'"))],
            i64(),
        )
    };
    let none = || call("zl_none", vec![], any());
    let pair = |a: Expr, b: Expr| call("zb_box_tuple", vec![list(vec![a, b], anys.clone())], any());
    // `debug.getupvalue(f, n)`: the name and value, or nothing.
    d.push(define(
        "zl_debug_getupvalue",
        &[&args],
        any(),
        vec![
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
            f.decl(call("zl_value_at", vec![args.e(), int(1)], any())),
            function_arg(&f, "setupvalue", 1),
            n.decl(int_arg(
                call("zl_value_at", vec![args.e(), int(2)], any()),
                "setupvalue",
                2,
            )),
            when(
                lt(len(args.e()), int(3)),
                vec![lua_error(text(
                    "bad argument #3 to 'setupvalue' (value expected)",
                ))],
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
            vec![lua_error(text(&format!(
                "bad argument #{number} to 'upvaluejoin' (invalid upvalue index)"
            )))],
        )
    };
    let lua_function = |f: Expr, number: usize| {
        when(
            lt(call("zl_func_id", vec![f], i64()), int(0)),
            vec![lua_error(text(&format!(
                "bad argument #{number} to 'upvaluejoin' (Lua function expected)"
            )))],
        )
    };
    let g = kept("g", any());
    let m = local("m", i64());
    d.push(define(
        "zl_debug_upvaluejoin",
        &[&args],
        unit(),
        vec![
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
            lua_function(f.e(), 1),
            lua_function(g.e(), 3),
            check_index(f.e(), n.e(), 2),
            check_index(g.e(), m.e(), 4),
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
                vec![lua_error(text(&format!(
                    "bad argument #1 to '{who}' (level out of range)"
                )))],
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
    let mut st = thread_arg(&args, &thread, &first);
    st.extend([
        y.decl(arg(int(1))),
        z.decl(nil()),
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
            vec![y.e(), text("bad argument #1 to 'getlocal'")],
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
        ret(pair(
            box_str(name.e()),
            at(
                call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
                i.e(),
            ),
        )),
    ]);
    d.push(define("zl_debug_getlocal", &[&args], any(), st));
    // `debug.setlocal([thread,] level, n, v)`: the name, or fail.
    let mut st = thread_arg(&args, &thread, &first);
    st.extend([
        y.decl(nil()),
        z.decl(nil()),
        level.decl(int_arg(arg(int(1)), "setlocal", 1)),
        n.decl(int_arg(arg(int(2)), "setlocal", 2)),
        when(
            lt(len(args.e()), add(first.e(), int(3))),
            vec![lua_error(text(
                "bad argument #3 to 'setlocal' (value expected)",
            ))],
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
        set_idx(
            call("zb_unbox_list_raw_any", vec![x.e()], anys.clone()),
            i.e(),
            arg(int(3)),
        ),
        expr(call("zl_dbg_mark_set", vec![depth.e(), i.e()], unit())),
        ret(box_str(name.e())),
    ]);
    d.push(define("zl_debug_setlocal", &[&args], any(), st));

    // ─── hooks ──────────────────────────────────────────────────
    // `debug.sethook([thread,] [hook, mask [, count]])`: no hook turns
    // hooks off.
    let mut st = thread_arg(&args, &thread, &first);
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
        f.decl(arg(int(1))),
        y.decl(arg(int(2))),
        z.decl(arg(int(3))),
        when(
            or(is_nil(f.e()), is_nil(y.e())),
            vec![
                set_global(HOOK_MASK, int(0)),
                set_global(HOOK_FN, nil()),
                set_global(HOOK_COUNT_EVERY, int(0)),
                ret_void(),
            ],
        ),
        function_arg(&f, "sethook", 1),
        name.decl(call(
            "zl_arg_str",
            vec![y.e(), text("bad argument #2 to 'sethook'")],
            string(),
        )),
        k.decl(int(0)),
        when(
            not(is_nil(z.e())),
            vec![k.set(int_arg(z.e(), "sethook", 3))],
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
        own_event(HOOK_RETURN, "return"),
        ret_void(),
    ]);
    d.push(define("zl_debug_sethook", &[&args], unit(), st));
    // `debug.gethook([thread])`: the hook, its mask and its count, or
    // fail when none is set.
    d.push(define(
        "zl_debug_gethook",
        &[&args],
        any(),
        vec![
            when(
                is_nil(read_global(HOOK_FN, any())),
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
    // A userdata's user values: this implementation makes no
    // userdata that has them.
    d.push(define(
        "zl_debug_getuservalue",
        &[&args],
        any(),
        vec![ret(pair(nil(), box_bool(bool(false))))],
    ));
    d.push(define(
        "zl_debug_setuservalue",
        &[&args],
        any(),
        vec![ret(call("zl_fail", vec![], any()))],
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
