//! Debug declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        table, tb, x, y, ..
    } = context;
    let cached = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            any(),
        )
    };
    let assign_global = |name: &str, value: Expr| {
        expr(node(
            zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                zyntax_typed_ast::typed_ast::TypedBinary {
                    op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                    left: Box::new(cached(name)),
                    right: Box::new(value),
                },
            ),
            any(),
        ))
    };

    // ─── debug ──────────────────────────────────────────────────
    d.push(define(
        "zl_debug_getmetatable",
        &[x],
        any(),
        vec![
            when(
                eq(category(x.e()), int(STR)),
                vec![ret(call("zl_string_metatable", vec![], any()))],
            ),
            when(
                is_file(x.e()),
                vec![ret(call("zl_file_metatable", vec![], any()))],
            ),
            when(
                not(is_table(x.e())),
                vec![
                    when(
                        is_userdata(x.e()),
                        vec![ret(call("zl_ud_meta", vec![x.e()], any()))],
                    ),
                    ret(call("zl_type_meta", vec![x.e()], any())),
                ],
            ),
            tb.decl(unbox_table(x.e(), t)),
            when(
                eq(meta_of(tb.e(), t), null(table.clone())),
                vec![ret(nil())],
            ),
            ret(box_table(meta_of(tb.e(), t))),
        ],
    ));
    d.push(define(
        "zl_debug_setmetatable",
        &[x, y],
        any(),
        vec![
            when(
                and(not(is_nil(y.e())), not(is_table(y.e()))),
                vec![lua_error(concat(vec![
                    text("bad argument #2 to 'setmetatable' (nil or table expected, got "),
                    arg_type_name(y.e()),
                    text(")"),
                ]))],
            ),
            when(
                eq(category(x.e()), int(STR)),
                vec![assign_global("zl_string_meta", y.e()), ret(x.e())],
            ),
            // A protected metatable is no protection from the debug
            // library.
            when(
                is_table(x.e()),
                vec![if_(
                    is_nil(y.e()),
                    vec![set_field(
                        unbox_table(x.e(), t),
                        "meta",
                        null(table.clone()),
                    )],
                    vec![set_field(
                        unbox_table(x.e(), t),
                        "meta",
                        unbox_table(y.e(), t),
                    )],
                )],
            ),
            when(
                is_userdata(x.e()),
                vec![
                    expr(call("zl_ud_set_meta", vec![x.e(), y.e()], unit())),
                    ret(x.e()),
                ],
            ),
            expr(call("zl_set_type_meta", vec![x.e(), y.e()], boolean())),
            ret(x.e()),
        ],
    ));
    d.push(global_var("zl_registry", any()));
    d.push(define(
        "zl_debug_getregistry",
        &[],
        any(),
        vec![
            when(
                is_nil(cached("zl_registry")),
                vec![assign_global(
                    "zl_registry",
                    box_table(call("zl_table_new", vec![], table.clone())),
                )],
            ),
            ret(cached("zl_registry")),
        ],
    ));
}
