//! Instance support for the Lua library.

use super::*;

/// What the library says about instances: how a table or a coroutine
/// prints, what `type` calls it, and that two are equal only when they
/// are the same object.
pub(super) fn instance_hooks(t: &Types) -> Vec<Decl> {
    let x = local("x", any());
    let a = local("a", any());
    let b = local("b", any());
    let code = local("code", i64());
    let p = local("p", usize());
    let addr = |x: Expr| call("zb_unbox_instance_raw", vec![x], i64());
    let hex = |x: Expr| call("zb_str_of_int_radix", vec![x, int32(16)], string());
    let mut d = vec![
        define(
            "zb_hook_instance_str",
            &[&x],
            string(),
            vec![
                when(
                    is_thread(x.e()),
                    vec![ret(add(text("thread: 0x"), hex(addr(x.e()))))],
                ),
                when(
                    is_file(x.e()),
                    vec![ret(call("zl_file_str", vec![x.e()], string()))],
                ),
                when(
                    is_table(x.e()),
                    vec![ret(add(text("table: 0x"), hex(addr(x.e()))))],
                ),
                when(
                    is_any_userdata(x.e()),
                    vec![ret(add(text("userdata: 0x"), hex(addr(x.e()))))],
                ),
                ret(add(text("table: 0x"), hex(addr(x.e())))),
            ],
        ),
        // Lua has no second spelling of a value.
        define(
            "zb_hook_instance_repr",
            &[&x],
            string(),
            vec![ret(call("zb_hook_instance_str", vec![x.e()], string()))],
        ),
        define(
            "zb_hook_instance_type",
            &[&x],
            string(),
            vec![
                when(is_table(x.e()), vec![ret(text("table"))]),
                when(is_thread(x.e()), vec![ret(text("thread"))]),
                when(is_file(x.e()), vec![ret(text("FILE*"))]),
                when(is_any_userdata(x.e()), vec![ret(text("userdata"))]),
                ret(text("table")),
            ],
        ),
        // Indexing, ordering and negation of an instance are Lua's own
        // operations, with their metamethods.
        define(
            "zb_hook_instance_getitem",
            &[&x, &b],
            any(),
            vec![ret(call("zl_index", vec![x.e(), b.e()], any()))],
        ),
        define(
            "zb_hook_instance_setitem",
            &[&x, &a, &b],
            unit(),
            vec![
                expr(call("zl_setindex", vec![x.e(), a.e(), b.e()], unit())),
                ret_void(),
            ],
        ),
        define(
            "zb_hook_instance_lt",
            &[&a, &b],
            boolean(),
            vec![ret(call("zl_lt", vec![a.e(), b.e()], boolean()))],
        ),
        define(
            "zb_hook_instance_le",
            &[&a, &b],
            boolean(),
            vec![ret(call("zl_le", vec![a.e(), b.e()], boolean()))],
        ),
        // Membership in a table is by iterating it, as for any value
        // without a membership test of its own.
        define(
            "zb_hook_instance_contains",
            &[&x, &a],
            boolean(),
            vec![ret(call(
                "zb_any_contains_iter",
                vec![x.e(), a.e()],
                boolean(),
            ))],
        ),
        define(
            "zb_hook_instance_unary",
            &[&code, &x],
            any(),
            vec![
                when(
                    eq(code.e(), int(0)),
                    vec![ret(call("zl_unm", vec![x.e()], any()))],
                ),
                lua_error(text("attempt to perform arithmetic on a table value")),
                ret(nil()),
            ],
        ),
        define(
            "zb_hook_instance_eq",
            &[&a, &b],
            boolean(),
            vec![ret(eq(addr(a.e()), addr(b.e())))],
        ),
        // Every table is true.
        define(
            "zb_hook_instance_truthy",
            &[&x],
            boolean(),
            vec![ret(bool(true))],
        ),
        define(
            "zb_hook_instance_hash",
            &[&x],
            i64(),
            vec![ret(addr(x.e()))],
        ),
        // Arithmetic on an instance is a metamethod, resolved by the
        // Lua arithmetic itself; the shared library never gets here.
        define(
            "zb_hook_instance_arith",
            &[&code, &a, &b],
            any(),
            vec![ret(call("zl_arith", vec![code.e(), a.e(), b.e()], any()))],
        ),
        define(
            "zb_hook_box_instance",
            &[&p],
            any(),
            vec![ret(call(
                "zb_box_instance",
                vec![cast(p.e(), i64()), int32(table_tag() as i32)],
                any(),
            ))],
        ),
        {
            let tag = local("tag", i32());
            define(
                "zb_hook_unbox_instance",
                &[&x, &tag],
                usize(),
                vec![
                    when(
                        ne(tag_of(x.e()), cast(tag.e(), i64())),
                        vec![lua_error(add(
                            text("expected a table, got "),
                            type_name(x.e()),
                        ))],
                    ),
                    ret(cast(addr(x.e()), usize())),
                ],
            )
        },
    ];
    // The shared library's shape hooks, which no Lua value reaches.
    let i = local("i", i64());
    let v = local("v", any());
    d.push(define(
        "zb_hook_shaped_items",
        &[&x],
        t.anys(),
        vec![ret(list(Vec::new(), t.anys()))],
    ));
    d.push(define(
        "zb_hook_shaped_get",
        &[&x, &i],
        any(),
        vec![ret(nil())],
    ));
    d.push(define(
        "zb_hook_shaped_set",
        &[&x, &i, &v],
        unit(),
        vec![ret_void()],
    ));
    d.push(define(
        "zb_hook_shaped_append",
        &[&x, &v],
        unit(),
        vec![ret_void()],
    ));
    d.push(define(
        "zb_hook_shaped_repr",
        &[&x],
        string(),
        vec![ret(text(""))],
    ));
    let ys = borrowed("ys", t.anys());
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let mask = local("mask", i64());
    d.push(define(
        "zb_hook_shaped_assign_slice",
        &[&x, &ys, &start, &stop, &step, &mask],
        unit(),
        vec![ret_void()],
    ));
    d.push(define(
        "zb_hook_shaped_getslice",
        &[&x, &start, &stop, &step, &mask],
        any(),
        vec![ret(nil())],
    ));
    d
}
