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
    vec![
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
    ]
}
