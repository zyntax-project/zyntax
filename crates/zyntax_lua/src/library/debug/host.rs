//! Host declarations.

use super::*;

pub(super) fn append(context: &Context, _t: &Types, d: &mut Vec<Decl>) {
    let Context { x, y, .. } = context;
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
    d.push(define("zl_type_meta", &[x], any(), type_meta(x.e())));
    // Gives `x`'s type the metatable `y`; whether `x` has such a type.
    d.push(define(
        "zl_set_type_meta",
        &[x, y],
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
}
