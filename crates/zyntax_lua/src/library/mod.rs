//! What Lua defines above the shared built-in library, as typed AST:
//! tables, the arithmetic and comparison of dynamic values under Lua's
//! rules, calls through function values with Lua's argument adjustment,
//! coroutines over fibers, and the standard library a program reaches
//! by name. Every function here is prefixed `zl_`; the shared library's
//! are `zb_`.
//!
//! Included by the crate and by its build script, which lowers the
//! library once for the snapshot the crate carries.

pub mod calls;
pub mod coroutines;
pub mod debug;
pub mod io;
pub mod patterns;
pub mod stdlib;
pub mod tables;
pub mod utf8;
pub mod values;

use zyntax_builtins::build::*;
use zyntax_typed_ast::type_registry::{FieldDef, TypeMetadata};
use zyntax_typed_ast::typed_ast::{TypedAnnotation, TypedClass, TypedDeclaration, TypedField};
use zyntax_typed_ast::{Mutability, Type, TypeId, Visibility};

/// The name the table struct is registered under.
pub const TABLE_TYPE: &str = "LuaTable";

/// The instance kinds this frontend boxes: a table's address and a
/// coroutine's fiber, each under [`zyntax_builtins::instance_tag`].
pub const TABLE_KIND: usize = 0;
pub const THREAD_KIND: usize = 1;
/// `error(nil)` in flight: nil means nothing pending, so a nil error
/// value travels as this instance and is nil again when taken.
pub const NIL_ERROR_KIND: usize = 2;
/// A file: the handle of one of the host's streams.
pub const FILE_KIND: usize = 3;
/// The error in flight while a suspended coroutine is closed: its
/// `<close>` handlers see nil, and nothing catches it on the way up.
pub const CLOSING_KIND: usize = 4;

pub fn table_tag() -> i64 {
    zyntax_builtins::instance_tag(TABLE_KIND)
}
pub fn thread_tag() -> i64 {
    zyntax_builtins::instance_tag(THREAD_KIND)
}
pub fn nil_error_tag() -> i64 {
    zyntax_builtins::instance_tag(NIL_ERROR_KIND)
}
pub fn file_tag() -> i64 {
    zyntax_builtins::instance_tag(FILE_KIND)
}
pub fn closing_tag() -> i64 {
    zyntax_builtins::instance_tag(CLOSING_KIND)
}
pub fn is_closing(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(closing_tag())))
}
pub fn closing_marker() -> Expr {
    call(
        "zb_box_instance_raw",
        vec![int(1), int32(closing_tag() as i32)],
        any(),
    )
}

/// The types the library is written against: `List<Any>` and the
/// table struct.
#[derive(Clone)]
pub struct Types {
    pub list_type: TypeId,
    pub table_type: TypeId,
}

impl Types {
    pub fn anys(&self) -> Type {
        zyntax_builtins::list_of(self.list_type, any())
    }
    pub fn table(&self) -> Type {
        table_ty(self.table_type)
    }
}

/// The table struct's type, a pointer to a heap object.
pub fn table_ty(id: TypeId) -> Type {
    Type::Named {
        id,
        type_args: Vec::new(),
        const_args: Vec::new(),
        variance: Vec::new(),
        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
    }
}

/// A table is an array part, a hash part and a metatable. The array
/// part is a boxed list of the values at keys `1..=n`, never holding a
/// trailing nil, so its length is a border. The hash part is a boxed
/// dict (see the shared library's dicts) or null until the first key
/// outside the array. The metatable is a table pointer or null. `high`
/// is the longest the array part has been before a store of nil
/// shortened it: a key up to it may have been removed mid-traversal,
/// which `next` continues from; one past it is no key of the table.
/// `shape` is the number of the shape a table was made with, 0 for
/// none: a shaped table carries its constant-key fields in typed slots
/// after this header, laid out by the program that made it, and
/// `present` has a bit set for each slot holding a value. The program
/// defines the hooks the library reads and writes them through.
const TABLE_FIELDS: [&str; 6] = ["arr", "hash", "meta", "high", "shape", "present"];

fn table_field_type(name: &str, id: TypeId) -> Type {
    match name {
        "meta" => table_ty(id),
        "high" | "shape" | "present" => i64(),
        _ => any(),
    }
}

/// The table header's fields, with their types.
pub fn table_header_fields(id: TypeId) -> Vec<(String, Type)> {
    TABLE_FIELDS
        .iter()
        .map(|name| (name.to_string(), table_field_type(name, id)))
        .collect()
}

/// Register the table struct in `registry` and return its id.
pub fn declare_table_type(registry: &mut zyntax_typed_ast::TypeRegistry) -> TypeId {
    let id = TypeId::next();
    declare_reference_struct(registry, id, TABLE_TYPE, &table_header_fields(id));
    id
}

/// Register a heap struct named `name` with `fields` under `id`.
pub fn declare_reference_struct(
    registry: &mut zyntax_typed_ast::TypeRegistry,
    id: TypeId,
    name: &str,
    fields: &[(String, Type)],
) {
    let fields: Vec<FieldDef> = fields
        .iter()
        .map(|(name, ty)| FieldDef {
            name: intern(name),
            ty: ty.clone(),
            visibility: Visibility::Public,
            mutability: Mutability::Mutable,
            is_static: false,
            span: SPAN,
            getter: None,
            setter: None,
            is_synthetic: false,
        })
        .collect();
    registry.register_type(zyntax_typed_ast::type_registry::TypeDefinition {
        id,
        module: None,
        name: intern(name),
        kind: zyntax_typed_ast::type_registry::TypeKind::Struct {
            fields: fields.clone(),
            is_tuple: false,
        },
        type_params: Vec::new(),
        constraints: Vec::new(),
        fields,
        methods: Vec::new(),
        constructors: Vec::new(),
        metadata: TypeMetadata {
            is_reference: true,
            ..Default::default()
        },
        span: SPAN,
    });
}

/// The struct's declaration, so the lowering lays it out.
pub fn table_class(id: TypeId) -> Decl {
    reference_struct_class(TABLE_TYPE, &table_header_fields(id))
}

/// A heap struct's declaration: every holder shares the one object.
pub fn reference_struct_class(name: &str, fields: &[(String, Type)]) -> Decl {
    typed_node_decl(TypedDeclaration::Class(TypedClass {
        name: intern(name),
        type_params: Vec::new(),
        extends: None,
        implements: Vec::new(),
        fields: fields
            .iter()
            .map(|(name, ty)| TypedField {
                name: intern(name),
                ty: ty.clone(),
                initializer: None,
                visibility: Visibility::Public,
                mutability: Mutability::Mutable,
                is_static: false,
                span: SPAN,
            })
            .collect(),
        methods: Vec::new(),
        constructors: Vec::new(),
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        // Every holder shares the one heap object.
        annotations: vec![TypedAnnotation {
            name: intern("reference"),
            args: Vec::new(),
            span: SPAN,
        }],
        span: SPAN,
    }))
}

fn typed_node_decl(d: TypedDeclaration) -> Decl {
    zyntax_typed_ast::TypedNode::new(d, Type::Unknown, SPAN)
}

// ─── shared helpers ─────────────────────────────────────────────────

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
/// The categories of the shared box, by number.
pub const BOOL: i64 = 1;
pub const INT: i64 = 2;
pub const UINT: i64 = 3;
pub const FLOAT: i64 = 4;
pub const STR: i64 = 5;
pub const CUSTOM: i64 = 255;

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
pub fn is_upvalue_id(x: Expr) -> Expr {
    and(
        ne(x.clone(), nil()),
        eq(
            tag_of(x),
            int(zyntax_builtins::instance_tag(debug::UPVALUE_ID_KIND)),
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
    when(
        and(
            not(is_func(h.clone())),
            is_nil(call("zl_meta_of", vec![h.clone(), text("__call")], any())),
        ),
        vec![lua_error(concat(vec![
            text("attempt to call a "),
            type_name(h),
            text(" value (metamethod '"),
            event,
            text("')"),
        ]))],
    )
}
fn is_nil_error(x: Expr) -> Expr {
    and(ne(x.clone(), nil()), eq(tag_of(x), int(nil_error_tag())))
}
fn nil_error() -> Expr {
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
pub fn arr_of(tb: Expr, t: &Types) -> Expr {
    call("zb_unbox_list_raw_any", vec![arr_field(tb)], t.anys())
}
pub fn hash_field(tb: Expr) -> Expr {
    fld(tb, "hash", any())
}
pub fn hash_of(tb: Expr, t: &Types) -> Expr {
    call("zb_unbox_list_raw_any", vec![hash_field(tb)], t.anys())
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

/// The hooks a program installs over its shapes, for the library's
/// generic paths: the slot a name has in a table's shape (-1 for
/// none), a slot's value boxed (nil when absent), a store into a slot
/// (0, or 1 when the value is not of the slot's kind), how many slots
/// a table's shape has, and a slot's name. Each is a function of the
/// program, reached through its code kept in a global: the library is
/// lowered before any program exists, and a chunk loaded later links
/// against neither. Installed by [`SHAPE_HOOKS_INSTALL`] before the
/// program runs; only a shaped table reaches them, and only a program
/// with shapes makes one.
pub const SHAPE_HOOKS: [&str; 5] = ["index", "load", "store", "count", "key"];
pub const SHAPE_HOOKS_INSTALL: &str = "zl_shape_hooks";

fn hook_global(hook: &str) -> String {
    format!("zl_hook_{hook}")
}

fn fn_type(params: &[Type], ret: Type) -> Type {
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

fn shape_hook_decls(t: &Types) -> Vec<Decl> {
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

/// A Lua error: reported through the hook, which never returns.
pub fn lua_error(message: Expr) -> Stmt {
    fatal("error", message)
}

/// A type error about `operand`, which the site describes: `(local
/// 'x')` and the like.
pub fn type_error(message: Expr, operand: Expr) -> Stmt {
    expr(call("zl_type_error", vec![message, operand], unit()))
}

/// What the library says about instances: how a table or a coroutine
/// prints, what `type` calls it, and that two are equal only when they
/// are the same object.
fn instance_hooks(t: &Types) -> Vec<Decl> {
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
                    is_upvalue_id(x.e()),
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
                when(is_thread(x.e()), vec![ret(text("thread"))]),
                when(is_file(x.e()), vec![ret(text("FILE*"))]),
                when(is_upvalue_id(x.e()), vec![ret(text("userdata"))]),
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

/// The error in flight, or nil. An error is raised by storing its value
/// here; every function that may raise leaves with a placeholder once
/// it sees the value set, until a `pcall` takes it.
pub const PENDING: &str = "zl_pending";
/// Whether the pending error is a stack overflow: a message handler
/// cannot run on one, as the reference has no stack left for it.
pub const OVERFLOWED: &str = "zl_overflowed";
/// How many times a message handler that raises is called again on
/// its own error before that is 'error in error handling'.
pub const HANDLER_RETRIES: i64 = 200;
/// The names of the chunks a program is made of besides the main one,
/// which `zl_chunk` names: a line stored in `zl_line` carries its
/// chunk's number in the bits above `LINE_BITS`, 0 for the main chunk
/// and `k` for the `k`th entry here.
pub const CHUNKS: &str = "zl_chunks";
/// The one empty array part every table without positional values
/// starts with, boxed; a write to a table's array part gives it one
/// of its own first (see `zl_arr_own`).
pub const ARR_EMPTY: &str = "zl_arr_empty";
/// How many program functions are on the stack; past `MAX_DEPTH` a
/// call is a stack overflow, as the reference's stack limit makes it.
/// A coroutine counts on from where it was resumed.
pub const DEPTH: &str = "zl_depth";
/// How many calls from the library into the program are in progress:
/// metamethods, iterators, comparators, what pcall protects, a
/// coroutine resumed. These nest on the reference's C stack, which
/// holds `CCALLS_LIMIT` of them.
pub const CCALLS: &str = "zl_ccalls";
pub const CCALLS_LIMIT: i64 = 200;
#[allow(dead_code)]
pub const MAX_DEPTH: i64 = 200_000;
pub const LINE_BITS: i64 = 32;
/// The globals table, set before the chunk runs when the program
/// reaches its globals through one.
pub const GLOBALS: &str = "zl_G";
/// The line of the statement running, for the position a message
/// carries; zero outside any.
pub const LINE: &str = "zl_line";
/// The chunk's name, for the same.
pub const CHUNK: &str = "zl_chunk";
/// Which operand a pending type error is about, for the variable
/// description the site appends: `OPERAND_LEFT` or `OPERAND_RIGHT`,
/// with `VARINFO_INSIDE` when the description goes after the message's
/// first word rather than at its end, and `VARINFO_CALL` when the
/// error is a call's, which only a call site describes; zero for any
/// other error. Set by every raise, cleared by the check that
/// consumes it.
pub const VARINFO: &str = "zl_varinfo";
/// Set by the conversion of a `for` limit: whether the limit lies past
/// the integers on the side the step moves away from, so the loop has
/// no iteration whatever its start.
pub const FOR_SKIP: &str = "zl_for_skip";
pub const OPERAND_LEFT: i64 = 1;
pub const OPERAND_RIGHT: i64 = 2;
pub const VARINFO_INSIDE: i64 = 4;
pub const VARINFO_CALL: i64 = 8;

pub fn global_var(name: &str, ty: Type) -> Decl {
    zyntax_typed_ast::TypedNode::new(
        TypedDeclaration::Variable(zyntax_typed_ast::typed_ast::TypedVariable {
            name: intern(name),
            ty,
            mutability: Mutability::Mutable,
            initializer: None,
            visibility: Visibility::Public,
        }),
        Type::Unknown,
        SPAN,
    )
}

pub fn read_global(name: &str, ty: Type) -> Expr {
    node(
        zyntax_typed_ast::typed_ast::TypedExpression::Variable(intern(name)),
        ty,
    )
}

pub fn set_global(name: &str, value: Expr) -> Stmt {
    let ty = value.ty.clone();
    expr(node(
        zyntax_typed_ast::typed_ast::TypedExpression::Binary(
            zyntax_typed_ast::typed_ast::TypedBinary {
                op: zyntax_typed_ast::typed_ast::BinaryOp::Assign,
                left: Box::new(read_global(name, ty.clone())),
                right: Box::new(value),
            },
        ),
        ty,
    ))
}

pub fn pending() -> Expr {
    read_global(PENDING, any())
}

/// An error was just raised: a program that keeps its call stack notes
/// where, for the report of an error nothing catches.
fn note_raise() -> Stmt {
    when(
        read_global(debug::DBG_ON, boolean()),
        vec![expr(call(
            "zl_dbg_note_raise",
            vec![read_global(LINE, i64())],
            unit(),
        ))],
    )
}

/// Raising: the first error stands until it is taken; a message gets
/// the running statement's position, an error value is kept as it is.
fn raising(t: &Types) -> Vec<Decl> {
    let kind = local("kind", string());
    let message = local("message", string());
    let v = kept("v", any());
    let s = local("s", string());
    let level = local("level", i64());
    let line = local("line", i64());
    let mut d = vec![
        global_var(PENDING, any()),
        global_var(LINE, i64()),
        global_var(CHUNK, string()),
        global_var(CHUNKS, any()),
        global_var(DEPTH, i64()),
        global_var(CCALLS, i64()),
        global_var(VARINFO, i64()),
        global_var(FOR_SKIP, boolean()),
        global_var(OVERFLOWED, boolean()),
        global_var(ARR_EMPTY, any()),
    ];
    // Entered below the floor: the error every deeper call would raise.
    d.push(define_cold(
        "zl_stack_overflow",
        &[],
        unit(),
        vec![
            set_global(OVERFLOWED, bool(true)),
            lua_error(text("stack overflow")),
            ret_void(),
        ],
    ));
    // The name of the chunk a stored line belongs to.
    let found = local("found", any());
    d.push(define(
        "zl_chunk_of",
        &[&line],
        string(),
        vec![
            when(
                or(
                    eq(shr(line.e(), int(LINE_BITS)), int(0)),
                    is_nil(read_global(CHUNKS, any())),
                ),
                vec![ret(read_global(CHUNK, string()))],
            ),
            // An internal table, read raw: nothing of the program's
            // runs while an error is positioned.
            found.decl(call(
                "zl_rawgeti",
                vec![
                    unbox_table(read_global(CHUNKS, any()), t),
                    shr(line.e(), int(LINE_BITS)),
                ],
                any(),
            )),
            when(is_nil(found.e()), vec![ret(read_global(CHUNK, string()))]),
            ret(get_str(found.e())),
        ],
    ));
    // A chunk a program runs besides its main one, under the number its
    // lines carry.
    let name = kept("name", string());
    let index = local("index", i64());
    d.push(define(
        "zl_chunk_add",
        &[&name, &index],
        unit(),
        vec![
            when(
                is_nil(read_global(CHUNKS, any())),
                vec![set_global(
                    CHUNKS,
                    box_table(call("zl_table_new", vec![], t.table())),
                )],
            ),
            expr(call(
                "zl_rawseti",
                vec![
                    unbox_table(read_global(CHUNKS, any()), t),
                    index.e(),
                    box_str(name.e()),
                ],
                unit(),
            )),
            ret_void(),
        ],
    ));
    // `message` positioned at a stored line: its chunk and line.
    d.push(define(
        "zl_position_at",
        &[&line, &message],
        string(),
        vec![
            when(le(line.e(), int(0)), vec![ret(message.e())]),
            ret(concat(vec![
                call("zl_chunk_of", vec![line.e()], string()),
                text(":"),
                call(
                    "zb_str_of_int",
                    vec![bitand(line.e(), int((1i64 << LINE_BITS) - 1))],
                    string(),
                ),
                text(": "),
                message.e(),
            ])),
        ],
    ));
    // The message with the position of the statement running.
    d.push(define(
        "zl_position",
        &[&message],
        string(),
        vec![ret(call(
            "zl_position_at",
            vec![read_global(LINE, i64()), message.e()],
            string(),
        ))],
    ));
    d.push(define_cold(
        "zl_raise_value",
        &[&v],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, int(0)),
            if_(
                is_nil(v.e()),
                vec![set_global(PENDING, nil_error())],
                vec![set_global(PENDING, v.e())],
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // The shared library's hook: a message, positioned.
    d.push(define_cold(
        "zb_hook_raise",
        &[&kind, &message],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, int(0)),
            set_global(
                PENDING,
                box_str(call("zl_position", vec![message.e()], string())),
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // A type error, positioned, with the operand it is about noted for
    // the site to describe.
    let operand = local("operand", i64());
    d.push(define_cold(
        "zl_type_error",
        &[&message, &operand],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            set_global(VARINFO, operand.e()),
            set_global(
                PENDING,
                box_str(call("zl_position", vec![message.e()], string())),
            ),
            note_raise(),
            ret_void(),
        ],
    ));
    // Calling nil a lookup found for a method or a field: the message
    // names it, as the reference's does. `how` is 0 for a method, 1 for
    // a field.
    let callee = local("callee", string());
    let how = local("how", i64());
    d.push(define_cold(
        "zl_raise_call_nil",
        &[&callee, &how],
        unit(),
        vec![
            lua_error(concat(vec![
                text("attempt to call a nil value ("),
                if_expr(eq(how.e(), int(0)), text("method '"), text("field '")),
                callee.e(),
                text("')"),
            ])),
            ret_void(),
        ],
    ));
    // The pending type error with the description of the operand it is
    // about, `left` or `right`, appended; the note is consumed. A
    // call's error is described at a call site only, and an
    // operator's only at an operator's.
    let left = kept("left", string());
    let right = kept("right", string());
    let calling = local("calling", boolean());
    let mark = local("mark", i64());
    let described = local("described", string());
    let tail = " has no integer representation";
    d.push(define_cold(
        "zl_annotate",
        &[&v, &left, &right, &calling],
        any(),
        vec![
            mark.decl(read_global(VARINFO, i64())),
            when(eq(mark.e(), int(0)), vec![ret(v.e())]),
            set_global(VARINFO, int(0)),
            when(
                ne(ne(bitand(mark.e(), int(VARINFO_CALL)), int(0)), calling.e()),
                vec![ret(v.e())],
            ),
            described.decl(if_expr(
                eq(bitand(mark.e(), int(3)), int(OPERAND_LEFT)),
                left.e(),
                right.e(),
            )),
            when(
                eq(call("zb_str_len", vec![described.e()], i64()), int(0)),
                vec![ret(v.e())],
            ),
            s.decl(get_str(v.e())),
            when(
                ne(bitand(mark.e(), int(VARINFO_INSIDE)), int(0)),
                vec![ret(box_str(concat(vec![
                    call(
                        "zb_str_substring",
                        vec![
                            s.e(),
                            int(0),
                            sub(
                                call("zb_str_len", vec![s.e()], i64()),
                                int(tail.len() as i64),
                            ),
                        ],
                        string(),
                    ),
                    described.e(),
                    text(tail),
                ])))],
            ),
            ret(box_str(add(s.e(), described.e()))),
        ],
    ));
    // `error(v, level)`: a string message at a level above zero is
    // positioned; anything else is the error value itself.
    d.push(define_cold(
        "zl_error",
        &[&v, &level],
        unit(),
        vec![
            if_(
                and(
                    and(not(is_nil(v.e())), eq(category(v.e()), int(STR))),
                    gt(level.e(), int(0)),
                ),
                vec![expr(call(
                    "zl_raise_value",
                    vec![box_str(call("zl_position", vec![get_str(v.e())], string()))],
                    unit(),
                ))],
                vec![expr(call("zl_raise_value", vec![v.e()], unit()))],
            ),
            ret_void(),
        ],
    ));
    // `error(v, 2)`: the position is the caller's, the line the
    // function was entered at.
    d.push(define_cold(
        "zl_error_at",
        &[&v, &line],
        unit(),
        vec![
            if_(
                and(
                    and(not(is_nil(v.e())), eq(category(v.e()), int(STR))),
                    gt(line.e(), int(0)),
                ),
                vec![expr(call(
                    "zl_raise_value",
                    vec![box_str(call(
                        "zl_position_at",
                        vec![line.e(), get_str(v.e())],
                        string(),
                    ))],
                    unit(),
                ))],
                vec![expr(call("zl_raise_value", vec![v.e()], unit()))],
            ),
            ret_void(),
        ],
    ));
    // The error taken by whoever handles it: the value, then nothing
    // pending.
    d.push(define(
        "zl_take_pending",
        &[],
        any(),
        vec![
            v.decl(pending()),
            set_global(PENDING, nil()),
            set_global(OVERFLOWED, bool(false)),
            when(is_nil_error(v.e()), vec![ret(nil())]),
            ret(v.e()),
        ],
    ));
    // An error nothing caught, reported by the host the way `lua`
    // reports it, ending the program with status 1. The host formats
    // the message itself, so the chunk's entry reaches nothing of the
    // library's `tostring`.
    d.push(extern_fn(
        "zl_report_uncaught",
        &[("err", any())],
        unit(),
        Some("$Lua$report_pending"),
    ));
    d.push(define_cold(
        "zl_report_pending",
        &[],
        unit(),
        vec![
            v.decl(pending()),
            when(is_nil(v.e()), vec![ret_void()]),
            set_global(PENDING, nil()),
            set_global(OVERFLOWED, bool(false)),
            expr(call("zl_report_uncaught", vec![v.e()], unit())),
            ret_void(),
        ],
    ));
    // The text an uncaught table error reports: what its `__tostring`
    // gives when that is a string, the error it raises if it raises one,
    // else nil. Reached only from the host, which compiles it when such
    // an error is reported.
    let h = local("h", any());
    let r = local("r", any());
    d.push(define_cold(
        ERROR_TEXT,
        &[&v],
        any(),
        vec![
            h.decl(call("zl_meta_of", vec![v.e(), text("__tostring")], any())),
            when(is_nil(h.e()), vec![ret(nil())]),
            r.decl(call(
                "zl_first",
                vec![call("zl_call_1", vec![h.e(), v.e()], any())],
                any(),
            )),
            when(
                not(is_nil(pending())),
                vec![ret(call("zl_take_pending", vec![], any()))],
            ),
            when(
                and(not(is_nil(r.e())), eq(category(r.e()), int(STR))),
                vec![ret(r.e())],
            ),
            ret(nil()),
        ],
    ));
    d
}

/// The library function giving an uncaught table error's text.
pub const ERROR_TEXT: &str = "zl_error_text";

/// The whole library for Lua: the shared library under Lua's
/// spellings, the table struct, and everything in this module.
pub fn library(policy: &zyntax_builtins::Policy) -> (zyntax_builtins::Library, Types) {
    let mut lib = zyntax_builtins::library(policy);
    let table_type = declare_table_type(&mut lib.type_registry);
    let t = Types {
        list_type: lib.list_type,
        table_type,
    };
    lib.declarations.push(table_class(table_type));
    lib.declarations.extend(shape_hook_decls(&t));
    lib.declarations.extend(instance_hooks(&t));
    lib.declarations.extend(raising(&t));
    lib.declarations.extend(tables::declarations(&t));
    lib.declarations.extend(values::declarations(policy, &t));
    lib.declarations.extend(calls::declarations(&t));
    lib.declarations.extend(coroutines::declarations(&t));
    lib.declarations.extend(patterns::declarations(&t));
    lib.declarations.extend(utf8::declarations(&t));
    lib.declarations.extend(io::declarations(&t));
    lib.declarations.extend(stdlib::declarations(policy, &t));
    lib.declarations.extend(debug::declarations(&t));
    lib.declarations.push(func_code_decl(&t));
    for d in &mut lib.declarations {
        if let TypedDeclaration::Function(f) = &mut d.node {
            f.annotations.push(strict_fp());
        }
    }
    lib.fallible = fallible_functions(&lib.declarations);
    (lib, t)
}

/// Every library function that may run the program's code before it
/// returns, through any number of calls: one that calls through a
/// code pointer, a local rather than a function's name, as the
/// callers of function values and the metamethod paths do.
pub fn reentrant_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    let declared: BTreeSet<String> = declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => f.name.resolve_global(),
            _ => None,
        })
        .collect();
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut reentrant = BTreeSet::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body)
        {
            let mut callees = BTreeSet::new();
            for s in &body.statements {
                callee_names(s, &mut callees);
            }
            if callees.iter().any(|c| !declared.contains(c)) {
                reentrant.insert(name.clone());
            }
            calls.insert(name, callees);
        }
    }
    loop {
        let before = reentrant.len();
        for (name, callees) in &calls {
            if callees.iter().any(|c| reentrant.contains(c)) {
                reentrant.insert(name.clone());
            }
        }
        if reentrant.len() == before {
            break;
        }
    }
    reentrant
}

/// The names called anywhere in a statement, block expressions and
/// all: what a lowered function calls.
pub fn callee_names(stmt: &Stmt, out: &mut std::collections::BTreeSet<String>) {
    use zyntax_typed_ast::typed_ast::{TypedExpression as E, TypedStatement as S};
    fn expr(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
        match &e.node {
            E::Call(c) => {
                if let E::Variable(n) = &c.callee.node
                    && let Some(name) = n.resolve_global()
                {
                    out.insert(name);
                } else {
                    expr(&c.callee, out);
                }
                for a in &c.positional_args {
                    expr(a, out);
                }
            }
            E::MethodCall(m) => {
                expr(&m.receiver, out);
                for a in &m.positional_args {
                    expr(a, out);
                }
            }
            E::Binary(b) => {
                expr(&b.left, out);
                expr(&b.right, out);
            }
            E::Unary(u) => expr(&u.operand, out),
            E::Index(i) => {
                expr(&i.object, out);
                expr(&i.index, out);
            }
            E::Field(f) => expr(&f.object, out),
            E::Cast(c) => expr(&c.expr, out),
            E::If(i) => {
                expr(&i.condition, out);
                expr(&i.then_branch, out);
                expr(&i.else_branch, out);
            }
            E::Array(items) | E::Tuple(items) => {
                for a in items {
                    expr(a, out);
                }
            }
            E::Struct(st) => {
                for f in &st.fields {
                    expr(&f.value, out);
                }
            }
            E::Block(b) => {
                for s in &b.statements {
                    callee_names(s, out);
                }
            }
            _ => {}
        }
    }
    match &stmt.node {
        S::Expression(e) => expr(e, out),
        S::Let(l) => {
            if let Some(init) = &l.initializer {
                expr(init, out);
            }
        }
        S::Return(Some(e)) => expr(e, out),
        S::If(i) => {
            expr(&i.condition, out);
            for s in &i.then_block.statements {
                callee_names(s, out);
            }
            if let Some(e) = &i.else_block {
                for s in &e.statements {
                    callee_names(s, out);
                }
            }
        }
        S::While(w) => {
            expr(&w.condition, out);
            for s in &w.body.statements {
                callee_names(s, out);
            }
        }
        S::Block(b) => {
            for s in &b.statements {
                callee_names(s, out);
            }
        }
        _ => {}
    }
}

/// `zl_func_id(f)`: the number of the program function a function
/// value is, or -1 for any other value, for a call site that knows
/// which functions a method may be and calls them directly. A
/// function of the program's making has its number after its code
/// and arity; the library's values carry no number there.
fn func_code_decl(t: &Types) -> Decl {
    let f = kept("f", any());
    let rec = borrowed("rec", t.anys());
    let slot = kept("slot", any());
    define(
        "zl_func_id",
        &[&f],
        i64(),
        vec![
            when(not(is_func(f.e())), vec![ret(int(-1))]),
            rec.decl(call("zb_unbox_list_raw_any", vec![f.e()], t.anys())),
            when(le(len(rec.e()), int(2)), vec![ret(int(-1))]),
            slot.decl(at(rec.e(), int(2))),
            when(
                or(is_nil(slot.e()), ne(category(slot.e()), int(INT))),
                vec![ret(int(-1))],
            ),
            ret(get_i64(slot.e())),
        ],
    )
}

/// Lua's floats round at every operation: no multiply and add of the
/// program's, or of the library's, is fused into one rounding.
pub fn strict_fp() -> TypedAnnotation {
    TypedAnnotation {
        name: intern("strict_fp"),
        args: Vec::new(),
        span: SPAN,
    }
}

/// Every function that raises, through any number of calls: one that
/// reaches the shared library's `zb_fatal`, or Lua's `zl_raise_value`
/// or `zl_type_error`.
fn fallible_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body)
        {
            let mut callees = BTreeSet::new();
            for s in &body.statements {
                callee_names(s, &mut callees);
            }
            calls.insert(name, callees);
        }
    }
    let mut fallible: BTreeSet<String> = [
        "zb_fatal".to_string(),
        "zl_raise_value".to_string(),
        "zl_type_error".to_string(),
    ]
    .into_iter()
    .collect();
    loop {
        let before = fallible.len();
        for (name, callees) in &calls {
            if callees.iter().any(|c| fallible.contains(c)) {
                fallible.insert(name.clone());
            }
        }
        if fallible.len() == before {
            break;
        }
    }
    fallible
}
