//! Wrappers declarations.

use super::*;

pub(super) fn append(context: &Context, t: &Types, d: &mut Vec<Decl>) {
    let Context {
        anys,
        args,
        i,
        n,
        out,
        table,
        tb,
        ..
    } = context;
    // `next` and `ipairs`'s iterator as function values, for `pairs`
    // and `ipairs` used as values rather than in a `for`.
    let env = borrowed("env", anys.clone());
    let code_of = |name: &str| {
        node(
            zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
            usize(),
        )
    };
    let builtin_value = |b: &Builtin| {
        let index = builtin_index(b);
        call(
            "zl_builtin_value",
            vec![
                int(index as i64),
                code_of(&wrapper_name(b)),
                int(VARIADIC_ARITY),
            ],
            any(),
        )
    };
    // ─── the value wrappers and the library tables ──────────────
    let packed = kept("packed", any());
    let method = local("method", boolean());
    for b in BUILTINS {
        // Each argument as the implementation takes it, from the
        // packed list; the result boxed. Whether the call is a method
        // call is taken first, so nothing the function calls sees it.
        let mut st = vec![
            method.decl(read_global(METHOD_CALL, boolean())),
            when(method.e(), vec![set_global(METHOD_CALL, bool(false))]),
            args.decl(call("zl_values", vec![packed.e()], anys.clone())),
        ];
        // Called as a value, the function is named as the reference
        // finds it in its library's table: `string.rep`. Called as a
        // method, by its name, not counting `self`.
        let qualified = if b.lib.is_empty() || HIDDEN_LIBS.contains(&b.lib) {
            b.name.to_string()
        } else {
            format!("{}.{}", b.lib, b.name)
        };
        let bad_arg = |n: usize, rest: &str| {
            let as_method = if n == 1 {
                format!("calling '{}' on bad self{rest}", b.name)
            } else {
                format!("bad argument #{} to '{}'{rest}", n - 1, b.name)
            };
            if_expr(
                method.e(),
                text(&as_method),
                text(&format!("bad argument #{n} to '{qualified}'{rest}")),
            )
        };
        // A required argument not passed at all.
        for (idx, p) in b.params.iter().enumerate() {
            let expected = match p {
                Param::Int | Param::Float => " (number expected, got no value)".to_string(),
                Param::Str | Param::Bytes => " (string expected, got no value)".to_string(),
                Param::Table => " (table expected, got no value)".to_string(),
                Param::Value => " (value expected)".to_string(),
                Param::Expected(kind) => format!(" ({kind} expected, got no value)"),
                _ => continue,
            };
            st.push(when(
                lt(len(args.e()), int(idx as i64 + 1)),
                vec![lua_error(bad_arg(idx + 1, &expected)), ret(nil())],
            ));
        }
        let mut call_args = Vec::new();
        for (idx, p) in b.params.iter().enumerate() {
            let arg = call("zl_value_at", vec![args.e(), int(idx as i64 + 1)], any());
            let what = bad_arg(idx + 1, "");
            call_args.push(match p {
                Param::Any | Param::Value | Param::Expected(_) => arg,
                Param::Int => call("zl_arg_int", vec![arg, what], i64()),
                Param::Float => call("zl_arg_float", vec![arg, what], f64()),
                Param::Str => call("zl_arg_str", vec![arg, what], string()),
                Param::Bytes => call("zl_arg_bytes", vec![arg, what], any()),
                Param::Table => call("zl_as_table", vec![arg, what], table.clone()),
                Param::OptInt(v) => call("zl_arg_opt_int", vec![arg, int(*v), what], i64()),
                Param::OptStr(v) => call("zl_arg_opt_str", vec![arg, text(v), what], string()),
                // Everything from here on; a slice past the end is empty.
                Param::Rest => call(
                    "zl_slice",
                    vec![args.e(), int(idx as i64), len(args.e())],
                    anys.clone(),
                ),
            });
        }
        let func = if b.params.contains(&Param::Bytes) {
            any_form(b.func)
        } else {
            b.func.to_string()
        };
        let result = call(&func, call_args, ret_type(b.ret, t));
        if b.lib == "debug" {
            st.push(set_global(debug::QUALIFY, bool(true)));
        }
        st.push(match b.ret {
            Ret::Unit => expr(result),
            Ret::Bool => ret(box_bool(result)),
            Ret::Int => ret(box_i64(result)),
            Ret::Float => ret(box_f64(result)),
            Ret::Str => ret(box_str(result)),
            Ret::Any | Ret::Multi => ret(result),
            Ret::Table => ret(box_table(result)),
        });
        if b.ret == Ret::Unit {
            st.push(ret(call("zl_none", vec![], any())));
        }
        d.push(define(&wrapper_name(b), &[&env, &packed], any(), st));
    }
    // Each library's table, built once: the wrappers as function
    // values, then the constants.
    for lib in LIBS.iter().chain(HIDDEN_LIBS) {
        let cache = format!("zl_lib_{lib}_cache");
        d.push(zyntax_typed_ast::TypedNode::new(
            zyntax_typed_ast::typed_ast::TypedDeclaration::Variable(
                zyntax_typed_ast::typed_ast::TypedVariable {
                    name: intern(&cache),
                    ty: any(),
                    mutability: zyntax_typed_ast::Mutability::Mutable,
                    initializer: None,
                    visibility: zyntax_typed_ast::Visibility::Public,
                },
            ),
            Type::Unknown,
            SPAN,
        ));
        let cached = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(&cache)),
                any(),
            )
        };
        // Cached before it is filled: `package.loaded` holds every
        // library's table, `package` included.
        let mut st = vec![
            when(not(is_nil(cached())), vec![ret(cached())]),
            tb.decl(call("zl_table_new", vec![], table.clone())),
            expr(node(
                zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                    zyntax_typed_ast::typed_ast::TypedBinary {
                        op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                        left: Box::new(cached()),
                        right: Box::new(box_table(tb.e())),
                    },
                ),
                any(),
            )),
        ];
        for b in BUILTINS.iter().filter(|b| b.lib == *lib) {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(b.name), builtin_value(b)],
                unit(),
            )));
        }
        for (_, name, c) in CONSTANTS.iter().filter(|(l, _, _)| l == lib) {
            let value = match c {
                Constant::Int(v) => box_i64(int(*v)),
                Constant::Float(v) => box_f64(float(*v)),
                Constant::Bytes(hex) => box_str(call("zl_bytes", vec![text(hex)], string())),
            };
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(name), value],
                unit(),
            )));
        }
        if *lib == "io" {
            for (name, getter) in [
                ("stdin", "zl_io_stdin"),
                ("stdout", "zl_io_stdout"),
                ("stderr", "zl_io_stderr"),
            ] {
                st.push(expr(call(
                    "zl_rawset_str",
                    vec![tb.e(), text(name), call(getter, vec![], any())],
                    unit(),
                )));
            }
        }
        if *lib == "package" {
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("loaded"),
                    call("zl_package_loaded", vec![], any()),
                ],
                unit(),
            )));
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("preload"),
                    call("zl_package_preload", vec![], any()),
                ],
                unit(),
            )));
            // `LUA_CPATH_5_4`, `LUA_CPATH` or the default, as the
            // host reads them.
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("cpath"),
                    box_str(call("zl_cpath_default", vec![], string())),
                ],
                unit(),
            )));
            // The searchers are built in; the table is there for a
            // program to check or replace, and must stay a table.
            st.push(expr(call(
                "zl_rawset_str",
                vec![
                    tb.e(),
                    text("searchers"),
                    box_table(call("zl_table_new", vec![], table.clone())),
                ],
                unit(),
            )));
        }
        st.push(ret(cached()));
        d.push(define(&lib_table_fn(lib), &[], any(), st));
    }
    // `arg`: the script's path at 0, its arguments from 1, built on
    // first use.
    d.push(extern_fn("zl_argc", &[], i64(), Some("$Lua$argc")));
    d.push(extern_fn(
        "zl_argv",
        &[("i", i64())],
        string(),
        Some("$Lua$argv"),
    ));
    d.push(zyntax_typed_ast::TypedNode::new(
        zyntax_typed_ast::typed_ast::TypedDeclaration::Variable(
            zyntax_typed_ast::typed_ast::TypedVariable {
                name: intern("zl_arg_cache"),
                ty: any(),
                mutability: zyntax_typed_ast::Mutability::Mutable,
                initializer: None,
                visibility: zyntax_typed_ast::Visibility::Public,
            },
        ),
        Type::Unknown,
        SPAN,
    ));
    {
        let cached = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern("zl_arg_cache")),
                any(),
            )
        };
        d.push(define(
            "zl_arg_table",
            &[],
            any(),
            vec![
                when(not(is_nil(cached())), vec![ret(cached())]),
                tb.decl(call("zl_table_new", vec![], table.clone())),
                n.decl(call("zl_argc", vec![], i64())),
                // The interpreter sits at -1, the script at 0.
                i.decl(int(0)),
                while_(
                    lt(i.e(), n.e()),
                    vec![
                        expr(call(
                            "zl_rawseti",
                            vec![
                                tb.e(),
                                sub(i.e(), int(1)),
                                box_str(call("zl_argv", vec![i.e()], string())),
                            ],
                            unit(),
                        )),
                        i.add_assign(int(1)),
                    ],
                ),
                expr(node(
                    zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                        zyntax_typed_ast::typed_ast::TypedBinary {
                            op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                            left: Box::new(cached()),
                            right: Box::new(box_table(tb.e())),
                        },
                    ),
                    any(),
                )),
                ret(cached()),
            ],
        ));
    }
    // The script's arguments after its name, as `...` at the main
    // chunk gives them.
    d.push(define(
        "zl_script_args",
        &[],
        anys.clone(),
        vec![
            out.decl(list(vec![], anys.clone())),
            n.decl(call("zl_argc", vec![], i64())),
            i.decl(int(2)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    push(out.e(), box_str(call("zl_argv", vec![i.e()], string()))),
                    i.add_assign(int(1)),
                ],
            ),
            ret(out.e()),
        ],
    ));
    // The globals table, for a program that reaches its globals through
    // `_G`: every base function, every library's table, `_G` itself,
    // `_VERSION` and `arg`. The program's own globals are set into it
    // as it runs.
    d.push(global_var(GLOBALS, table.clone()));
    {
        // Built once, and reachable through `zl_G` while it is being
        // filled: `package.loaded` holds `_G`.
        let g = || {
            node(
                zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(GLOBALS)),
                table.clone(),
            )
        };
        let mut st = vec![
            when(ne(g(), null(table.clone())), vec![ret(g())]),
            tb.decl(call("zl_table_new", vec![], table.clone())),
            expr(node(
                zyntax_typed_ast::typed_ast::TypedExpression::Binary(
                    zyntax_typed_ast::typed_ast::TypedBinary {
                        op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                        left: Box::new(g()),
                        right: Box::new(tb.e()),
                    },
                ),
                table.clone(),
            )),
        ];
        for b in BUILTINS.iter().filter(|b| b.lib.is_empty()) {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(b.name), builtin_value(b)],
                unit(),
            )));
        }
        for lib in LIBS {
            st.push(expr(call(
                "zl_rawset_str",
                vec![tb.e(), text(lib), call(&lib_table_fn(lib), vec![], any())],
                unit(),
            )));
        }
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_G"), box_table(tb.e())],
            unit(),
        )));
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("_VERSION"), box_str(text(LUA_VERSION))],
            unit(),
        )));
        st.push(expr(call(
            "zl_rawset_str",
            vec![tb.e(), text("arg"), call("zl_arg_table", vec![], any())],
            unit(),
        )));
        st.push(ret(tb.e()));
        d.push(define("zl_globals_table", &[], table.clone(), st));
    }
    // The metatable every string shares, `{__index = string}`, which a
    // program may edit; a member of a string is what its `__index`
    // says.
    d.push(global_var("zl_string_meta", any()));
}
