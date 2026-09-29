//! Foreign objects: values of the program that embeds the runtime,
//! boxed under [`FOREIGN_TAG`] with a word of the embedder's. Every
//! operation on one is the embedder's, through the `$Foreign$*` symbols
//! (`zyntax_embed::foreign`); an error it reports is raised here as a
//! library error of the kind it names.

use crate::build::*;
use crate::{FOREIGN_TAG, list_of};
use zyntax_typed_ast::TypeId;

/// Whether `x` is a foreign object.
pub fn is_foreign(x: Expr) -> Expr {
    and(
        ne(x.clone(), null(any())),
        eq(
            cast(call("zb_box_tag", vec![x], i32()), i64()),
            int(FOREIGN_TAG),
        ),
    )
}

pub(crate) fn declarations(list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let items = local("items", anys.clone());
    let n = local("n", i64());
    let i = local("i", i64());
    let x = local("x", any());
    let a = local("a", any());
    let b = local("b", any());
    let v = local("v", any());
    let name = local("name", string());
    let args = borrowed("args", anys.clone());
    let r = local("r", any());
    let rf = local("rf", f64());
    let ok = local("ok", i32());
    let key = local("key", i64());
    let kind = local("kind", string());
    // The error the embedder reported, raised as the library's own.
    let raise_reported = || {
        block_of(vec![
            kind.decl(call("zb_foreign_error_kind", vec![], string())),
            when(
                ne(kind.e(), null(string())),
                vec![expr(call(
                    "zb_fatal",
                    vec![kind.e(), call("zb_foreign_error_message", vec![], string())],
                    unit(),
                ))],
            ),
        ])
    };
    let data = || fld(args.e(), "data", i64());
    let len = || fld(args.e(), "len", i64());
    let mut declarations = vec![
        extern_fn(
            "zb_foreign_get_raw",
            &[("x", any()), ("name", string())],
            any(),
            Some("$Foreign$get"),
        ),
        extern_fn(
            "zb_foreign_set_raw",
            &[("x", any()), ("name", string()), ("v", any())],
            unit(),
            Some("$Foreign$set"),
        ),
        extern_fn(
            "zb_foreign_get_float_raw",
            &[("x", any()), ("name", string())],
            f64(),
            Some("$Foreign$get_float"),
        ),
        extern_fn(
            "zb_foreign_set_float_raw",
            &[("x", any()), ("name", string()), ("v", f64())],
            i32(),
            Some("$Foreign$set_float"),
        ),
        extern_fn(
            "zb_foreign_get_float_key_raw",
            &[("x", any()), ("key", i64())],
            f64(),
            Some("$Foreign$get_float_key"),
        ),
        extern_fn(
            "zb_foreign_set_float_key_raw",
            &[("x", any()), ("key", i64()), ("v", f64())],
            i32(),
            Some("$Foreign$set_float_key"),
        ),
        extern_fn(
            "zb_foreign_call_raw",
            &[("x", any()), ("data", i64()), ("len", i64())],
            any(),
            Some("$Foreign$call"),
        ),
        extern_fn(
            "zb_foreign_invoke_raw",
            &[
                ("x", any()),
                ("name", string()),
                ("data", i64()),
                ("len", i64()),
            ],
            any(),
            Some("$Foreign$invoke"),
        ),
        extern_fn(
            "zb_foreign_import_raw",
            &[("name", string())],
            any(),
            Some("$Foreign$import"),
        ),
        extern_fn(
            "zb_foreign_eq_raw",
            &[("a", any()), ("b", any())],
            i32(),
            Some("$Foreign$eq"),
        ),
        // `x` as text, and the name of its type.
        extern_fn(
            "zb_foreign_str",
            &[("x", any())],
            string(),
            Some("$Foreign$str"),
        ),
        extern_fn(
            "zb_foreign_type",
            &[("x", any())],
            string(),
            Some("$Foreign$type"),
        ),
        // How many values `x` is when it is several given at once, -1
        // when it is not; then value `i` of them.
        extern_fn(
            "zb_foreign_values_len",
            &[("x", any())],
            i64(),
            Some("$Foreign$values_len"),
        ),
        extern_fn(
            "zb_foreign_value",
            &[("x", any()), ("i", i64())],
            any(),
            Some("$Foreign$value"),
        ),
        // The length of the bytes `x` is a buffer of, -1 when none.
        extern_fn(
            "zb_foreign_bytes_len",
            &[("x", any())],
            i64(),
            Some("$Foreign$bytes_len"),
        ),
        extern_fn(
            "zb_foreign_hash",
            &[("x", any())],
            i64(),
            Some("$Foreign$hash"),
        ),
        // The kind of the error the last operation reported, null when
        // none; then its message, which clears it.
        extern_fn(
            "zb_foreign_error_kind",
            &[],
            string(),
            Some("$Foreign$error_kind"),
        ),
        extern_fn(
            "zb_foreign_error_message",
            &[],
            string(),
            Some("$Foreign$error_message"),
        ),
        define(
            "zb_is_foreign",
            &[&x],
            boolean(),
            vec![ret(is_foreign(x.e()))],
        ),
        // The member `name` of `x`: a field's value, or a method as a
        // value that takes its receiver first.
        define(
            "zb_foreign_get",
            &[&x, &name],
            any(),
            vec![
                r.decl(call("zb_foreign_get_raw", vec![x.e(), name.e()], any())),
                raise_reported(),
                ret(r.e()),
            ],
        ),
        define(
            "zb_foreign_set",
            &[&x, &name, &v],
            unit(),
            vec![
                expr(call(
                    "zb_foreign_set_raw",
                    vec![x.e(), name.e(), v.e()],
                    unit(),
                )),
                raise_reported(),
                ret_void(),
            ],
        ),
        define(
            "zb_foreign_get_float",
            &[&x, &name],
            f64(),
            vec![
                rf.decl(call(
                    "zb_foreign_get_float_raw",
                    vec![x.e(), name.e()],
                    f64(),
                )),
                when(eq(rf.e(), float(f64::MAX)), vec![raise_reported()]),
                ret(rf.e()),
            ],
        ),
        define(
            "zb_foreign_set_float",
            &[&x, &name, &rf],
            unit(),
            vec![
                ok.decl(call(
                    "zb_foreign_set_float_raw",
                    vec![x.e(), name.e(), rf.e()],
                    i32(),
                )),
                when(eq(ok.e(), int32(0)), vec![raise_reported()]),
                ret_void(),
            ],
        ),
        define(
            "zb_foreign_get_float_key",
            &[&x, &key],
            f64(),
            vec![
                rf.decl(call(
                    "zb_foreign_get_float_key_raw",
                    vec![x.e(), key.e()],
                    f64(),
                )),
                when(eq(rf.e(), float(f64::MAX)), vec![raise_reported()]),
                ret(rf.e()),
            ],
        ),
        define(
            "zb_foreign_set_float_key",
            &[&x, &key, &rf],
            unit(),
            vec![
                ok.decl(call(
                    "zb_foreign_set_float_key_raw",
                    vec![x.e(), key.e(), rf.e()],
                    i32(),
                )),
                when(eq(ok.e(), int32(0)), vec![raise_reported()]),
                ret_void(),
            ],
        ),
        // What the embedder gave: several values given at once as the
        // program's tuple of them, anything else as it is.
        define(
            "zb_foreign_values",
            &[&x],
            any(),
            vec![
                when(not(is_foreign(x.e())), vec![ret(x.e())]),
                n.decl(call("zb_foreign_values_len", vec![x.e()], i64())),
                when(lt(n.e(), int(0)), vec![ret(x.e())]),
                items.decl(list(vec![], anys.clone())),
                i.decl(int(0)),
                while_(
                    lt(i.e(), n.e()),
                    vec![
                        expr(mcall(
                            items.e(),
                            "push",
                            vec![call("zb_foreign_value", vec![x.e(), i.e()], any())],
                            unit(),
                        )),
                        i.add_assign(int(1)),
                    ],
                ),
                ret(call("zb_box_tuple", vec![items.e()], any())),
            ],
        ),
        define(
            "zb_foreign_call",
            &[&x, &args],
            any(),
            vec![
                r.decl(call(
                    "zb_foreign_call_raw",
                    vec![x.e(), data(), len()],
                    any(),
                )),
                raise_reported(),
                ret(call("zb_foreign_values", vec![r.e()], any())),
            ],
        ),
        // The method `name` of `x`, called with `args`.
        define(
            "zb_foreign_invoke",
            &[&x, &name, &args],
            any(),
            vec![
                r.decl(call(
                    "zb_foreign_invoke_raw",
                    vec![x.e(), name.e(), data(), len()],
                    any(),
                )),
                raise_reported(),
                ret(call("zb_foreign_values", vec![r.e()], any())),
            ],
        ),
        // The embedder's module `name`, or None when it has none.
        define(
            "zb_foreign_import",
            &[&name],
            any(),
            vec![
                r.decl(call("zb_foreign_import_raw", vec![name.e()], any())),
                raise_reported(),
                ret(r.e()),
            ],
        ),
        // The same, raising ModuleNotFoundError when there is none.
        define(
            "zb_foreign_module",
            &[&name],
            any(),
            vec![
                r.decl(call("zb_foreign_import_raw", vec![name.e()], any())),
                raise_reported(),
                when(
                    eq(r.e(), null(any())),
                    vec![fatal(
                        "ModuleNotFoundError",
                        add(add(text("No module named '"), name.e()), text("'")),
                    )],
                ),
                ret(r.e()),
            ],
        ),
        define(
            "zb_foreign_eq",
            &[&a, &b],
            boolean(),
            vec![ret(ne(
                call("zb_foreign_eq_raw", vec![a.e(), b.e()], i32()),
                int32(0),
            ))],
        ),
    ];

    // Calls whose arity is known use fixed parameters all the way into the
    // embedder. Dynamic and variadic callers keep the list-based entries.
    for arity in 0..=crate::functions::MAX_CALL_ARITY {
        let call_raw = format!("zb_foreign_call_fixed_raw_{arity}");
        let invoke_raw = format!("zb_foreign_invoke_fixed_raw_{arity}");
        let call_name = format!("zb_foreign_call_{arity}");
        let invoke_name = format!("zb_foreign_invoke_{arity}");
        let call_symbol = format!("$Foreign$call{arity}");
        let invoke_symbol = format!("$Foreign$invoke{arity}");
        let fixed_names: Vec<&'static str> = (0..arity)
            .map(|i| Box::leak(format!("a{i}").into_boxed_str()) as &'static str)
            .collect();
        let fixed: Vec<Local> = fixed_names.iter().map(|name| local(name, any())).collect();

        let mut call_extern_params = vec![("x", any())];
        call_extern_params.extend(fixed_names.iter().map(|name| (*name, any())));
        declarations.push(extern_fn(
            &call_raw,
            &call_extern_params,
            any(),
            Some(&call_symbol),
        ));
        let mut invoke_extern_params = vec![("x", any()), ("name", string())];
        invoke_extern_params.extend(fixed_names.iter().map(|name| (*name, any())));
        declarations.push(extern_fn(
            &invoke_raw,
            &invoke_extern_params,
            any(),
            Some(&invoke_symbol),
        ));

        let mut call_params: Vec<&Local> = vec![&x];
        call_params.extend(&fixed);
        let mut call_args = vec![x.e()];
        call_args.extend(fixed.iter().map(Local::e));
        declarations.push(define(
            &call_name,
            &call_params,
            any(),
            vec![
                r.decl(call(&call_raw, call_args, any())),
                raise_reported(),
                ret(call("zb_foreign_values", vec![r.e()], any())),
            ],
        ));

        let mut invoke_params: Vec<&Local> = vec![&x, &name];
        invoke_params.extend(&fixed);
        let mut invoke_args = vec![x.e(), name.e()];
        invoke_args.extend(fixed.iter().map(Local::e));
        declarations.push(define(
            &invoke_name,
            &invoke_params,
            any(),
            vec![
                r.decl(call(&invoke_raw, invoke_args, any())),
                raise_reported(),
                ret(call("zb_foreign_values", vec![r.e()], any())),
            ],
        ));

        let call_float_raw = format!("zb_foreign_call_float_raw_{arity}");
        let invoke_float_raw = format!("zb_foreign_invoke_float_raw_{arity}");
        let call_float_name = format!("zb_foreign_call_float_{arity}");
        let invoke_float_name = format!("zb_foreign_invoke_float_{arity}");
        let call_float_symbol = format!("$Foreign$call_float{arity}");
        let invoke_float_symbol = format!("$Foreign$invoke_float{arity}");
        let invoke_float_key_raw = format!("zb_foreign_invoke_float_key_raw_{arity}");
        let invoke_float_key_name = format!("zb_foreign_invoke_float_key_{arity}");
        let invoke_float_key_symbol = format!("$Foreign$invoke_float_key{arity}");
        let floats: Vec<Local> = fixed_names.iter().map(|name| local(name, f64())).collect();

        let mut call_float_extern_params = vec![("x", any())];
        call_float_extern_params.extend(fixed_names.iter().map(|name| (*name, f64())));
        declarations.push(extern_fn(
            &call_float_raw,
            &call_float_extern_params,
            f64(),
            Some(&call_float_symbol),
        ));
        let mut invoke_float_extern_params = vec![("x", any()), ("name", string())];
        invoke_float_extern_params.extend(fixed_names.iter().map(|name| (*name, f64())));
        declarations.push(extern_fn(
            &invoke_float_raw,
            &invoke_float_extern_params,
            f64(),
            Some(&invoke_float_symbol),
        ));
        let mut invoke_float_key_extern_params = vec![("x", any()), ("key", i64())];
        invoke_float_key_extern_params.extend(fixed_names.iter().map(|name| (*name, f64())));
        declarations.push(extern_fn(
            &invoke_float_key_raw,
            &invoke_float_key_extern_params,
            f64(),
            Some(&invoke_float_key_symbol),
        ));

        let mut call_float_params: Vec<&Local> = vec![&x];
        call_float_params.extend(&floats);
        let mut call_float_args = vec![x.e()];
        call_float_args.extend(floats.iter().map(Local::e));
        declarations.push(define(
            &call_float_name,
            &call_float_params,
            f64(),
            vec![
                rf.decl(call(&call_float_raw, call_float_args, f64())),
                when(eq(rf.e(), float(f64::MAX)), vec![raise_reported()]),
                ret(rf.e()),
            ],
        ));

        let mut invoke_float_params: Vec<&Local> = vec![&x, &name];
        invoke_float_params.extend(&floats);
        let mut invoke_float_args = vec![x.e(), name.e()];
        invoke_float_args.extend(floats.iter().map(Local::e));
        declarations.push(define(
            &invoke_float_name,
            &invoke_float_params,
            f64(),
            vec![
                rf.decl(call(&invoke_float_raw, invoke_float_args, f64())),
                when(eq(rf.e(), float(f64::MAX)), vec![raise_reported()]),
                ret(rf.e()),
            ],
        ));

        let mut invoke_float_key_params: Vec<&Local> = vec![&x, &key];
        invoke_float_key_params.extend(&floats);
        let mut invoke_float_key_args = vec![x.e(), key.e()];
        invoke_float_key_args.extend(floats.iter().map(Local::e));
        declarations.push(define(
            &invoke_float_key_name,
            &invoke_float_key_params,
            f64(),
            vec![
                rf.decl(call(&invoke_float_key_raw, invoke_float_key_args, f64())),
                when(eq(rf.e(), float(f64::MAX)), vec![raise_reported()]),
                ret(rf.e()),
            ],
        ));
    }

    declarations
}
