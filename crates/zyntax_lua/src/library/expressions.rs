//! Expressions support for the Lua library.

use super::*;

pub fn node(e: zyntax_typed_ast::typed_ast::TypedExpression, ty: Type) -> Expr {
    zyntax_typed_ast::TypedNode::new(e, ty, SPAN)
}

pub fn category(x: Expr) -> Expr {
    call("zb_any_category", vec![x], i64())
}

pub fn tag_of(x: Expr) -> Expr {
    cast(call("zb_box_tag", vec![x], i32()), i64())
}

pub fn box_i64(v: Expr) -> Expr {
    call("zb_box_i64", vec![v], any())
}

pub fn box_f64(v: Expr) -> Expr {
    call("zb_box_f64", vec![v], any())
}

pub fn box_bool(v: Expr) -> Expr {
    call("zb_box_bool", vec![v], any())
}

pub fn box_str(v: Expr) -> Expr {
    call("zb_box_str", vec![v], any())
}

pub fn get_i64(x: Expr) -> Expr {
    call("zb_box_payload_i64", vec![x], i64())
}

pub fn get_f64(x: Expr) -> Expr {
    call("zb_box_payload_f64", vec![x], f64())
}

pub fn get_bool(x: Expr) -> Expr {
    ne(call("zb_box_payload_bool", vec![x], i32()), int32(0))
}

pub fn get_str(x: Expr) -> Expr {
    call("zb_box_get_str", vec![x], string())
}

pub fn nil() -> Expr {
    null(any())
}

pub fn is_nil(x: Expr) -> Expr {
    eq(x, nil())
}

pub fn is_cat(cat: &Local, c: i64) -> Expr {
    eq(cat.e(), int(c))
}

pub fn is_int_cat(cat: &Local) -> Expr {
    or(is_cat(cat, INT), is_cat(cat, UINT))
}

pub fn is_number_cat(cat: &Local) -> Expr {
    or(is_int_cat(cat), is_cat(cat, FLOAT))
}

/// Whether a box holds a table: the custom category under the table's
/// kind.
pub fn is_table(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(table_tag())))
}

pub fn is_thread(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(thread_tag())))
}

pub fn is_file(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(file_tag())))
}

/// A light userdata.
pub fn is_light(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(light_tag())))
}

/// A full userdata.
pub fn is_userdata(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(userdata_tag())))
}

/// A full or a light userdata.
pub fn is_any_userdata(x: Expr) -> Expr {
    and(
        ne(x.clone(), nil()),
        or(
            eq(tag_of(x.clone()), int(userdata_tag())),
            eq(tag_of(x), int(light_tag())),
        ),
    )
}

pub fn is_func(x: Expr) -> Expr {
    and(
        ne(x.clone(), nil()),
        eq(tag_of(x), int(zyntax_builtins::FUNC_TAG)),
    )
}

/// `h`, a metamethod for `event`, checked to be callable before it is
/// called: the error names the event, as the reference's does.
pub fn metamethod_call_check(h: Expr, event: Expr) -> Stmt {
    block_of(vec![
        when(
            and(
                not(is_func(h.clone())),
                is_nil(call("zl_meta_of", vec![h.clone(), text("__call")], any())),
            ),
            vec![lua_error(concat(vec![
                text("attempt to call a "),
                obj_type_name(h),
                text(" value (metamethod '"),
                event.clone(),
                text("')"),
            ]))],
        ),
        metamethod_site(event),
    ])
}

/// The call about to be made is of the metamethod for `event`
/// (`index`, `add`, ...): a program that keeps its call stack names the
/// frame after it.
pub fn metamethod_site(event: Expr) -> Stmt {
    when(
        read_global(debug::DBG_ON, boolean()),
        vec![set_global(
            debug::DBG_SITE,
            call("zl_dbg_mm_site", vec![event], i64()),
        )],
    )
}

pub fn is_nil_error(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(nil_error_tag())))
}

pub(super) fn nil_error() -> Expr {
    // Any payload but zero, which boxes to nil.
    call(
        "zb_box_instance_raw",
        vec![int(1), int32(nil_error_tag() as i32)],
        any(),
    )
}

/// The start of a `luaL_argerror` message: the argument's number and
/// the function's name.
pub fn bad_arg(n: usize, name: &str) -> Expr {
    text(&format!("bad argument #{n} to '{name}'"))
}

/// The same for argument `i + 1`, `i` a value.
pub fn bad_arg_at(i: Expr, name: &str) -> Expr {
    concat(vec![
        text("bad argument #"),
        call("zb_str_of_int", vec![add(i, int(1))], string()),
        text(&format!(" to '{name}'")),
    ])
}

/// The table a box holds; the box is known to hold one.
pub fn unbox_table(x: Expr, t: &Types) -> Expr {
    cast(call("zb_unbox_instance_raw", vec![x], i64()), t.table())
}

/// A table as a dynamic value.
pub fn box_table(tb: Expr) -> Expr {
    call(
        "zb_box_instance",
        vec![cast(tb, i64()), int32(table_tag() as i32)],
        any(),
    )
}

pub fn arr_field(tb: Expr) -> Expr {
    fld(tb, "arr", any())
}

/// A flat array part: the elements of one number kind, stored without
/// a box each.
#[derive(Clone, Copy)]
pub enum Flat {
    Int,
    Float,
}

impl Flat {
    pub const ALL: [Flat; 2] = [Flat::Int, Flat::Float];

    fn kind(self) -> zyntax_builtins::Kind {
        match self {
            Flat::Int => zyntax_builtins::Kind::Int,
            Flat::Float => zyntax_builtins::Kind::Float,
        }
    }

    /// The box tag of the list.
    pub fn tag(self) -> i64 {
        self.kind().list_tag()
    }

    /// Whether the array part is flat of this kind.
    pub fn is(self, tb: Expr) -> Expr {
        eq(
            cast(call("zb_box_header_tag", vec![arr_field(tb)], i32()), i64()),
            int(self.tag()),
        )
    }

    pub fn elem(self) -> Type {
        match self {
            Flat::Int => i64(),
            Flat::Float => f64(),
        }
    }

    pub fn list_ty(self, t: &Types) -> Type {
        match self {
            Flat::Int => t.ints(),
            Flat::Float => t.floats(),
        }
    }

    /// The array part as a list of this kind; it is known to be one.
    pub fn list(self, tb: Expr, t: &Types) -> Expr {
        let unbox = match self {
            Flat::Int => "zb_unbox_list_raw_i64",
            Flat::Float => "zb_unbox_list_raw_f64",
        };
        call(unbox, vec![arr_field(tb)], self.list_ty(t))
    }

    /// A list of this kind as an array part.
    pub fn box_list(self, list: Expr) -> Expr {
        let name = match self {
            Flat::Int => "zb_list_box_i64",
            Flat::Float => "zb_list_box_f64",
        };
        call(name, vec![list], any())
    }

    /// An element as a dynamic value.
    pub fn boxed(self, e: Expr) -> Expr {
        match self {
            Flat::Int => box_i64(e),
            Flat::Float => box_f64(e),
        }
    }

    /// The element a dynamic value of this kind holds.
    pub fn unboxed(self, v: Expr) -> Expr {
        match self {
            Flat::Int => get_i64(v),
            Flat::Float => get_f64(v),
        }
    }

    /// The category of a dynamic value of this kind.
    pub fn category(self) -> i64 {
        match self {
            Flat::Int => INT,
            Flat::Float => FLOAT,
        }
    }
}

/// Whether the array part is flat of either kind.
pub fn arr_is_flat(tb: Expr) -> Expr {
    or(Flat::Int.is(tb.clone()), Flat::Float.is(tb))
}

/// The array part viewed as a list of boxes without a check. Only its
/// length is meaningful when it may be flat.
pub fn arr_raw(tb: Expr, t: &Types) -> Expr {
    call("zb_unbox_list_raw_any", vec![arr_field(tb)], t.anys())
}

/// The array part as a list of boxes, made so first when it is flat.
pub fn arr_of(tb: Expr, t: &Types) -> Expr {
    call("zl_arr_generic", vec![tb], t.anys())
}

pub fn hash_field(tb: Expr) -> Expr {
    fld(tb, "hash", any())
}

pub fn hash_of(tb: Expr, t: &Types) -> Expr {
    call("zb_dict_raw", vec![hash_field(tb)], t.dict())
}

pub fn meta_of(tb: Expr, t: &Types) -> Expr {
    fld(tb, "meta", t.table())
}

pub fn high_of(tb: Expr) -> Expr {
    fld(tb, "high", i64())
}

pub fn shape_of(tb: Expr) -> Expr {
    fld(tb, "shape", i64())
}

pub fn is_shaped(tb: Expr) -> Expr {
    ne(shape_of(tb), int(0))
}

/// Whether slot `i` of a shaped table holds a value: its bit of the
/// header's `present` word.
pub fn slot_present(tb: Expr, i: Expr) -> Expr {
    ne(bitand(shr(fld(tb, "present", i64()), i), int(1)), int(0))
}

pub(super) fn hook_global(hook: &str) -> String {
    format!("zl_hook_{hook}")
}

pub(super) fn fn_type(params: &[Type], ret: Type) -> Type {
    use zyntax_typed_ast::type_registry::{
        AsyncKind, CallingConvention, NullabilityKind, ParamInfo,
    };
    Type::Function {
        params: params
            .iter()
            .map(|ty| ParamInfo {
                name: None,
                ty: ty.clone(),
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            })
            .collect(),
        return_type: Box::new(ret),
        is_varargs: false,
        has_named_params: false,
        has_default_params: false,
        async_kind: AsyncKind::Sync,
        calling_convention: CallingConvention::Default,
        nullability: NullabilityKind::NonNull,
    }
}

pub(super) fn shape_hook_decls(t: &Types) -> Vec<Decl> {
    let table = t.table();
    let tb = kept("t", table.clone());
    let name = kept("name", string());
    let i = local("i", i64());
    let v = kept("v", any());
    let signatures: [(&str, Vec<&Local>, Type); 5] = [
        ("index", vec![&tb, &name], i64()),
        ("load", vec![&tb, &i], any()),
        ("store", vec![&tb, &i, &v], i64()),
        ("count", vec![&tb], i64()),
        ("key", vec![&tb, &i], string()),
    ];
    let mut d = Vec::new();
    for (hook, params, ret_ty) in &signatures {
        let global = hook_global(hook);
        let code_ty = fn_type(
            &params.iter().map(|p| p.ty.clone()).collect::<Vec<_>>(),
            ret_ty.clone(),
        );
        let unbox = format!("zl_hook_{hook}_code");
        let fp = local("fp", code_ty.clone());
        d.push(global_var(&global, any()));
        d.push(extern_fn(
            &unbox,
            &[("x", any())],
            code_ty.clone(),
            Some("zyntax_box_pointer"),
        ));
        d.push(define(
            &format!("zl_shape_{hook}"),
            params,
            ret_ty.clone(),
            vec![
                fp.decl(call(&unbox, vec![read_global(&global, any())], code_ty)),
                ret(call(
                    "fp",
                    params.iter().map(|p| p.e()).collect(),
                    ret_ty.clone(),
                )),
            ],
        ));
    }
    let codes: Vec<Local> = SHAPE_HOOKS
        .iter()
        .map(|hook| local(Box::leak(hook.to_string().into_boxed_str()), usize()))
        .collect();
    d.push(define(
        SHAPE_HOOKS_INSTALL,
        &codes.iter().collect::<Vec<_>>(),
        unit(),
        SHAPE_HOOKS
            .iter()
            .zip(&codes)
            .map(|(hook, code)| {
                set_global(
                    &hook_global(hook),
                    call(
                        "zb_box_fnptr_raw",
                        vec![code.e(), int32(zyntax_builtins::CODE_TAG as i32)],
                        any(),
                    ),
                )
            })
            .chain(std::iter::once(ret_void()))
            .collect(),
    ));
    d
}

pub fn set_field(tb: Expr, name: &str, value: Expr) -> Stmt {
    let ty = value.ty.clone();
    expr(node(
        zyntax_typed_ast::typed_ast::TypedExpression::Binary(
            zyntax_typed_ast::typed_ast::TypedBinary {
                op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                left: Box::new(fld(tb, name, ty.clone())),
                right: Box::new(value),
            },
        ),
        ty,
    ))
}

pub fn len(xs: Expr) -> Expr {
    mcall(xs, "len", vec![], i64())
}

pub fn push(xs: Expr, v: Expr) -> Stmt {
    expr(mcall(xs, "push", vec![v], unit()))
}

pub fn at(xs: Expr, i: Expr) -> Expr {
    idx(xs, i, any())
}

pub fn str_eq(a: Expr, b: Expr) -> Expr {
    call("zb_str_eq", vec![a, b], boolean())
}

pub fn concat(parts: Vec<Expr>) -> Expr {
    let mut it = parts.into_iter();
    let first = it.next().expect("something to join");
    it.fold(first, add)
}

pub fn type_name(x: Expr) -> Expr {
    call("zb_any_type", vec![x], string())
}

/// The type a runtime error names (`luaT_objtypename`): a table's
/// metatable's `__name` when it is a string, else the type's name.
pub fn obj_type_name(x: Expr) -> Expr {
    call("zl_objtypename", vec![x], string())
}

/// The type a bad argument's `got` names (`luaL_typeerror`): the
/// metatable's `__name` when it is a string, then `light userdata`,
/// then the type's name.
pub fn arg_type_name(x: Expr) -> Expr {
    call("zl_argtypename", vec![x], string())
}

/// A Lua error: reported through the hook, which never returns.
pub fn lua_error(message: Expr) -> Stmt {
    fatal("error", message)
}

/// A type error about `operand`, which the site describes: `(local
/// 'x')` and the like.
pub fn type_error(message: Expr, operand: Expr) -> Stmt {
    expr(call("zl_type_error", vec![message, operand], unit()))
}
