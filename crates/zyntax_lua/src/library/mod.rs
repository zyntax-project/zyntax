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

pub fn table_tag() -> i64 {
    zyntax_builtins::instance_tag(TABLE_KIND)
}
pub fn thread_tag() -> i64 {
    zyntax_builtins::instance_tag(THREAD_KIND)
}
pub fn nil_error_tag() -> i64 {
    zyntax_builtins::instance_tag(NIL_ERROR_KIND)
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
/// outside the array. The metatable is a table pointer or null.
const TABLE_FIELDS: [&str; 3] = ["arr", "hash", "meta"];

fn table_field_type(name: &str, id: TypeId) -> Type {
    match name {
        "meta" => table_ty(id),
        _ => any(),
    }
}

/// Register the table struct in `registry` and return its id.
pub fn declare_table_type(registry: &mut zyntax_typed_ast::TypeRegistry) -> TypeId {
    let id = TypeId::next();
    let fields: Vec<FieldDef> = TABLE_FIELDS
        .iter()
        .map(|name| FieldDef {
            name: intern(name),
            ty: table_field_type(name, id),
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
        name: intern(TABLE_TYPE),
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
    id
}

/// The struct's declaration, so the lowering lays it out.
pub fn table_class(id: TypeId) -> Decl {
    typed_node_decl(TypedDeclaration::Class(TypedClass {
        name: intern(TABLE_TYPE),
        type_params: Vec::new(),
        extends: None,
        implements: Vec::new(),
        fields: TABLE_FIELDS
            .iter()
            .map(|name| TypedField {
                name: intern(name),
                ty: table_field_type(name, id),
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

pub fn arr_of(tb: Expr, t: &Types) -> Expr {
    call(
        "zb_unbox_list_raw_any",
        vec![fld(tb, "arr", any())],
        t.anys(),
    )
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
                ret(add(text("table: 0x"), hex(addr(x.e())))),
            ],
        ),
        define(
            "zb_hook_instance_type",
            &[&x],
            string(),
            vec![
                when(is_thread(x.e()), vec![ret(text("thread"))]),
                ret(text("table")),
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
    d
}

/// The error in flight, or nil. An error is raised by storing its value
/// here; every function that may raise leaves with a placeholder once
/// it sees the value set, until a `pcall` takes it.
pub const PENDING: &str = "zl_pending";
/// The globals table, set before the chunk runs when the program
/// reaches its globals through one.
pub const GLOBALS: &str = "zl_G";
/// The line of the statement running, for the position a message
/// carries; zero outside any.
pub const LINE: &str = "zl_line";
/// The chunk's name, for the same.
pub const CHUNK: &str = "zl_chunk";

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

/// Raising: the first error stands until it is taken; a message gets
/// the running statement's position, an error value is kept as it is.
fn raising() -> Vec<Decl> {
    let kind = local("kind", string());
    let message = local("message", string());
    let v = kept("v", any());
    let level = local("level", i64());
    let mut d = vec![
        global_var(PENDING, any()),
        global_var(LINE, i64()),
        global_var(CHUNK, string()),
    ];
    // The message with the position of the statement running.
    d.push(define(
        "zl_position",
        &[&message],
        string(),
        vec![
            when(le(read_global(LINE, i64()), int(0)), vec![ret(message.e())]),
            ret(concat(vec![
                read_global(CHUNK, string()),
                text(":"),
                call("zb_str_of_int", vec![read_global(LINE, i64())], string()),
                text(": "),
                message.e(),
            ])),
        ],
    ));
    d.push(define_cold(
        "zl_raise_value",
        &[&v],
        unit(),
        vec![
            when(not(is_nil(pending())), vec![ret_void()]),
            if_(
                is_nil(v.e()),
                vec![set_global(PENDING, nil_error())],
                vec![set_global(PENDING, v.e())],
            ),
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
            set_global(
                PENDING,
                box_str(call("zl_position", vec![message.e()], string())),
            ),
            ret_void(),
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
    let line = local("line", i64());
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
                    vec![box_str(concat(vec![
                        read_global(CHUNK, string()),
                        text(":"),
                        call("zb_str_of_int", vec![line.e()], string()),
                        text(": "),
                        get_str(v.e()),
                    ]))],
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
            when(is_nil_error(v.e()), vec![ret(nil())]),
            ret(v.e()),
        ],
    ));
    // An error nothing caught, reported the way `lua` reports it, ending
    // the program with status 1.
    d.push(define_cold(
        "zl_report_pending",
        &[],
        unit(),
        vec![
            when(is_nil(pending()), vec![ret_void()]),
            expr(call(
                "zb_eprintln",
                vec![add(
                    text("lua: "),
                    call(
                        "zl_tostring",
                        vec![call("zl_take_pending", vec![], any())],
                        string(),
                    ),
                )],
                unit(),
            )),
            expr(call("zb_exit", vec![int32(1)], unit())),
            ret_void(),
        ],
    ));
    d
}

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
    lib.declarations.extend(instance_hooks(&t));
    lib.declarations.extend(raising());
    lib.declarations.extend(tables::declarations(&t));
    lib.declarations.extend(values::declarations(policy, &t));
    lib.declarations.extend(calls::declarations(&t));
    lib.declarations.extend(coroutines::declarations(&t));
    lib.declarations.extend(patterns::declarations(&t));
    lib.declarations.extend(utf8::declarations(&t));
    lib.declarations.extend(stdlib::declarations(policy, &t));
    lib.fallible = fallible_functions(&lib.declarations);
    (lib, t)
}

/// Every function that raises, through any number of calls: one that
/// reaches the shared library's `zb_fatal`, or Lua's `zl_raise_value`.
fn fallible_functions(declarations: &[Decl]) -> std::collections::BTreeSet<String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut calls: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for d in declarations {
        if let TypedDeclaration::Function(f) = &d.node
            && let (Some(name), Some(body)) = (f.name.resolve_global(), &f.body)
        {
            let mut callees = BTreeSet::new();
            for s in &body.statements {
                callees_of_stmt(s, &mut callees);
            }
            calls.insert(name, callees);
        }
    }
    let mut fallible: BTreeSet<String> = ["zb_fatal".to_string(), "zl_raise_value".to_string()]
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
