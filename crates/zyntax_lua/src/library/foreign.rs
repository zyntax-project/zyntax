//! Foreign objects: values of the program that embeds the runtime, which
//! Lua sees as userdata. Indexing one with a string reads or writes the
//! embedder's member of that name, and calling one calls it, so
//! `o:m(...)` is the index giving the method, called with `o` first, as
//! for a table. `require` asks the embedder for a module after Lua's own
//! searchers (see `zl_require`).

use super::*;
use zyntax_builtins::functions::VARIADIC_ARITY;

/// The function value every call of a foreign object goes through.
const CALLER: &str = "zl_foreign_caller_value";

pub(super) fn declarations(t: &Types) -> Vec<Decl> {
    let anys = t.anys();
    let o = kept("o", any());
    let k = kept("k", any());
    let v = kept("v", any());
    let env = borrowed("env", anys.clone());
    let packed = kept("packed", any());
    let vals = local("vals", anys.clone());
    let rest = local("rest", anys.clone());
    let i = local("i", i64());
    let mut d = Vec::new();

    let is_name = |k: &Local| and(not(is_nil(k.e())), eq(category(k.e()), int(STR)));
    let not_a_name = |k: &Local| {
        lua_error(concat(vec![
            text("attempt to index a userdata value with a "),
            type_name(k.e()),
            text(" key"),
        ]))
    };
    d.push(define(
        "zl_foreign_index",
        &[&o, &k],
        any(),
        vec![
            when(
                is_name(&k),
                vec![ret(call(
                    "zb_foreign_get",
                    vec![o.e(), get_str(k.e())],
                    any(),
                ))],
            ),
            not_a_name(&k),
            ret(nil()),
        ],
    ));
    d.push(define(
        "zl_foreign_setindex",
        &[&o, &k, &v],
        unit(),
        vec![
            when(
                is_name(&k),
                vec![
                    expr(call(
                        "zb_foreign_set",
                        vec![o.e(), get_str(k.e()), v.e()],
                        unit(),
                    )),
                    ret_void(),
                ],
            ),
            not_a_name(&k),
            ret_void(),
        ],
    ));

    // The caller's code: the object, then the arguments it is called
    // with, as `zl_callee` puts them.
    d.push(define(
        "zl_foreign_call_code",
        &[&env, &packed],
        any(),
        vec![
            vals.decl(call("zl_values", vec![packed.e()], anys.clone())),
            rest.decl(list(vec![], anys.clone())),
            i.decl(int(1)),
            while_(
                lt(i.e(), len(vals.e())),
                vec![push(rest.e(), at(vals.e(), i.e())), i.add_assign(int(1))],
            ),
            ret(call(
                "zb_foreign_call",
                vec![at(vals.e(), int(0)), rest.e()],
                any(),
            )),
        ],
    ));
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    d.push(global_var(CALLER, any()));
    d.push(define(
        "zl_foreign_caller",
        &[],
        any(),
        vec![
            when(
                is_nil(read_global(CALLER, any())),
                vec![set_global(
                    CALLER,
                    call(
                        "zb_func_new",
                        vec![
                            code_of("zl_foreign_call_code"),
                            int(VARIADIC_ARITY),
                            list(vec![], anys.clone()),
                        ],
                        any(),
                    ),
                )],
            ),
            ret(read_global(CALLER, any())),
        ],
    ));
    d
}
