//! Lists: the `List<T>` type, and the rules every language layers on the
//! type's own operations (`push`, `pop_last`, `insert_at`, `remove_at`,
//! `len`, indexing): negative indexes, errors, search, ordering,
//! printing, slicing. Built once per element kind.

use crate::build::*;
use crate::{Kind, Policy, TUPLE_TAG, list_of};
use zyntax_typed_ast::type_registry::{FieldDef, TypeMetadata, TypeParam, Variance};
use zyntax_typed_ast::typed_ast::{TypedClass, TypedDeclaration, TypedField, TypedTypeParam};
use zyntax_typed_ast::typed_builder::TypedASTBuilder;
use zyntax_typed_ast::{Mutability, Type, TypeId, TypedNode, Visibility};

/// Register `struct List<T> { data, len, capacity }` and return its id.
pub(crate) fn declare_list_type(b: &mut TypedASTBuilder) -> TypeId {
    let field = |name: &str| FieldDef {
        name: intern(name),
        ty: i64(),
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: SPAN,
        getter: None,
        setter: None,
        is_synthetic: false,
    };
    b.registry.register_struct_type(
        intern("List"),
        vec![TypeParam {
            name: intern("T"),
            bounds: Vec::new(),
            variance: Variance::Invariant,
            default: None,
            span: SPAN,
            is_const: false,
            const_ty: None,
        }],
        vec![field("data"), field("len"), field("capacity")],
        Vec::new(),
        Vec::new(),
        TypeMetadata::default(),
        SPAN,
    )
}

/// The struct declaration itself, so the lowering lays it out.
fn list_class() -> Decl {
    let field = |name: &str| TypedField {
        name: intern(name),
        ty: i64(),
        initializer: None,
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: SPAN,
    };
    TypedNode::new(
        TypedDeclaration::Class(TypedClass {
            name: intern("List"),
            type_params: vec![TypedTypeParam {
                name: intern("T"),
                bounds: Vec::new(),
                default: None,
                span: SPAN,
                is_const: false,
                const_ty: None,
            }],
            extends: None,
            implements: Vec::new(),
            fields: vec![field("data"), field("len"), field("capacity")],
            methods: Vec::new(),
            constructors: Vec::new(),
            visibility: Visibility::Public,
            is_abstract: false,
            is_final: false,
            annotations: Vec::new(),
            span: SPAN,
        }),
        Type::Unknown,
        SPAN,
    )
}

type Binary = Box<dyn Fn(Expr, Expr) -> Expr>;
type Unary = Box<dyn Fn(Expr) -> Expr>;

/// How one element kind compares, prints, boxes and reads back.
struct KindOps {
    /// The library's own kinds; `None` for a tuple shape a frontend
    /// registered.
    kind: Option<Kind>,
    /// The suffix of this kind's functions.
    suffix: String,
    /// The tag a boxed list of this kind carries.
    tag: i64,
    elem: Type,
    list: Type,
    /// `List<String>`, for building text piece by piece.
    strs: Type,
    eq: Binary,
    lt: Binary,
    repr: Unary,
    /// An element as a dynamic value.
    boxed: Unary,
    /// A dynamic value read back as an element, or a TypeError.
    read: Unary,
}

fn ops(kind: Kind, list_type: TypeId) -> KindOps {
    let elem = kind.ty();
    let (eq, lt, repr): (Binary, Binary, Unary) = match kind {
        Kind::Int => (
            Box::new(|a, b| eq(a, b)),
            Box::new(|a, b| lt(a, b)),
            Box::new(|x| call("zb_str_of_int", vec![x], string())),
        ),
        Kind::Float => (
            Box::new(|a, b| eq(a, b)),
            Box::new(|a, b| lt(a, b)),
            Box::new(|x| call("zb_float_repr", vec![x], string())),
        ),
        Kind::Str => (
            Box::new(|a, b| call("zb_str_eq", vec![a, b], boolean())),
            Box::new(|a, b| call("zb_str_lt", vec![a, b], boolean())),
            Box::new(|x| call("zb_str_repr", vec![x], string())),
        ),
        // The same box is equal to itself before its value is looked
        // at, which is how a membership test or an element comparison
        // treats identity; shared boxes of small integers make that the
        // common case.
        Kind::Any => (
            Box::new(|a, b| {
                or(
                    eq(a.clone(), b.clone()),
                    call("zb_any_eq", vec![a, b], boolean()),
                )
            }),
            Box::new(|a, b| call("zb_any_lt", vec![a, b], boolean())),
            Box::new(|x| call("zb_any_repr", vec![x], string())),
        ),
        // Instances compare by identity; ordering them is an error, and
        // printing one goes through its boxed form.
        Kind::Ptr => (
            Box::new(|a, b| eq(a, b)),
            Box::new(|a, b| call("zb_ptr_lt", vec![a, b], boolean())),
            Box::new(|x| {
                call(
                    "zb_any_repr",
                    vec![call("zb_hook_box_instance", vec![x], any())],
                    string(),
                )
            }),
        ),
        // Array storage prints as the number it reads as.
        Kind::F32 => (
            Box::new(|a, b| eq(a, b)),
            Box::new(|a, b| lt(a, b)),
            Box::new(|x| call("zb_float_repr", vec![cast(x, f64())], string())),
        ),
        _ => (
            Box::new(|a, b| eq(a, b)),
            Box::new(|a, b| lt(a, b)),
            Box::new(|x| call("zb_str_of_int", vec![cast(x, i64())], string())),
        ),
    };
    // A primitive boxes as itself on the push; an instance address
    // boxes as the instance; array storage boxes as the number it
    // reads as. Reading back checks the kind; an instance list takes
    // the class tag its elements must carry, which the generated
    // function reads from its own parameter; array storage takes the
    // number through its range check.
    let boxed: Unary = match kind {
        Kind::Ptr => Box::new(|e| call("zb_hook_box_instance", vec![e], any())),
        Kind::F32 => Box::new(|e| cast(e, f64())),
        k if k.is_narrow_int() => Box::new(|e| cast(e, i64())),
        _ => Box::new(|e| e),
    };
    let read: Unary = match kind {
        Kind::Int => Box::new(|e| call("zb_any_as_i64", vec![e], i64())),
        Kind::Float => Box::new(|e| call("zb_any_as_f64", vec![e], f64())),
        Kind::Str => Box::new(|e| call("zb_any_as_str", vec![e], string())),
        Kind::Ptr => Box::new(|e| {
            call(
                "zb_hook_unbox_instance",
                vec![e, local("tag", i32()).e()],
                usize(),
            )
        }),
        Kind::Any => Box::new(|e| e),
        Kind::F32 => Box::new(|e| cast(call("zb_any_as_f64", vec![e], f64()), Kind::F32.ty())),
        k => Box::new(move |e| {
            call(
                &format!("zb_list_narrow_{}", k.suffix()),
                vec![call("zb_any_as_i64", vec![e], i64())],
                k.ty(),
            )
        }),
    };
    KindOps {
        kind: Some(kind),
        suffix: kind.suffix().to_string(),
        tag: kind.list_tag(),
        list: list_of(list_type, elem.clone()),
        strs: list_of(list_type, string()),
        elem,
        eq,
        lt,
        repr,
        boxed,
        read,
    }
}

/// How a field of a tuple shape is compared, printed, boxed and read
/// back: what a frontend knows of the element type it put there.
#[derive(Clone, Debug)]
pub enum Field {
    Int,
    Float,
    Bool,
    Str,
    /// A dynamic value.
    Any,
    /// A dict or a set, stored as the box its tag identifies (a header
    /// has an identity a copy inside the tuple would lose) and read raw
    /// as `ty`, the list of dynamic values.
    Dict {
        ty: Type,
    },
    Set {
        ty: Type,
    },
    /// An instance of the frontend's class carrying `tag`, held by
    /// address as `ty`.
    Instance {
        ty: Type,
        tag: i32,
    },
    /// A list of the kind whose functions carry `suffix`, stored as the
    /// box of the kind's tag and read raw as `ty`.
    List {
        suffix: String,
        ty: Type,
    },
    /// A typed array stored as the kind whose functions carry `suffix`,
    /// boxed under `tag`, which carries its typecode `letter`.
    Array {
        suffix: String,
        ty: Type,
        tag: i64,
        letter: String,
    },
    /// A tuple of the shape whose functions carry `suffix`.
    Tuple {
        suffix: String,
        ty: Type,
    },
}

impl Field {
    fn ty(&self) -> Type {
        match self {
            Field::Int => i64(),
            Field::Float => f64(),
            Field::Bool => boolean(),
            Field::Str => string(),
            Field::Any
            | Field::Dict { .. }
            | Field::Set { .. }
            | Field::List { .. }
            | Field::Array { .. } => any(),
            Field::Instance { ty, .. } | Field::Tuple { ty, .. } => ty.clone(),
        }
    }

    /// The header a stored dict, set or list box holds; the tag was
    /// checked when the field was stored.
    fn raw(&self, x: Expr) -> Expr {
        match self {
            Field::Dict { ty } | Field::Set { ty } => {
                call("zb_unbox_list_raw_any", vec![x], ty.clone())
            }
            Field::List { suffix, ty } | Field::Array { suffix, ty, .. } => {
                call(&format!("zb_unbox_list_raw_{suffix}"), vec![x], ty.clone())
            }
            _ => x,
        }
    }

    fn eq(&self, a: Expr, b: Expr) -> Expr {
        match self {
            Field::Int | Field::Float | Field::Bool => eq(a, b),
            Field::Str => call("zb_str_eq", vec![a, b], boolean()),
            Field::Any => or(
                eq(a.clone(), b.clone()),
                call("zb_any_eq", vec![a, b], boolean()),
            ),
            Field::Dict { .. } => call("zb_dict_eq", vec![self.raw(a), self.raw(b)], boolean()),
            Field::Set { .. } => call("zb_set_eq", vec![self.raw(a), self.raw(b)], boolean()),
            Field::Instance { .. } => eq(cast(a, usize()), cast(b, usize())),
            Field::List { suffix, .. } | Field::Array { suffix, .. } => call(
                &format!("zb_list_eq_{suffix}"),
                vec![self.raw(a), self.raw(b)],
                boolean(),
            ),
            Field::Tuple { suffix, .. } => {
                call(&format!("zb_tuple_eq_{suffix}"), vec![a, b], boolean())
            }
        }
    }

    fn lt(&self, a: Expr, b: Expr) -> Expr {
        match self {
            Field::Int | Field::Float => lt(a, b),
            Field::Bool => lt(cast(a, i64()), cast(b, i64())),
            Field::Str => call("zb_str_lt", vec![a, b], boolean()),
            Field::Any | Field::Dict { .. } | Field::Set { .. } => {
                call("zb_any_lt", vec![a, b], boolean())
            }
            Field::Instance { .. } => call(
                "zb_ptr_lt",
                vec![cast(a, usize()), cast(b, usize())],
                boolean(),
            ),
            Field::List { suffix, .. } | Field::Array { suffix, .. } => call(
                &format!("zb_list_lt_{suffix}"),
                vec![self.raw(a), self.raw(b)],
                boolean(),
            ),
            Field::Tuple { suffix, .. } => {
                call(&format!("zb_tuple_lt_{suffix}"), vec![a, b], boolean())
            }
        }
    }

    fn repr(&self, x: Expr) -> Expr {
        match self {
            Field::Int => call("zb_str_of_int", vec![x], string()),
            Field::Float => call("zb_float_repr", vec![x], string()),
            Field::Bool => call("zb_bool_repr", vec![x], string()),
            Field::Str => call("zb_str_repr", vec![x], string()),
            Field::Any => call("zb_any_repr", vec![x], string()),
            Field::Dict { .. } => call("zb_dict_repr", vec![self.raw(x)], string()),
            Field::Set { .. } => call("zb_set_repr", vec![self.raw(x)], string()),
            Field::Instance { .. } => call("zb_any_repr", vec![self.boxed(x)], string()),
            Field::List { suffix, .. } => call(
                &format!("zb_list_repr_{suffix}"),
                vec![self.raw(x)],
                string(),
            ),
            // `array('i', [1, 2])`, and `array('i')` when empty.
            Field::Array { suffix, letter, .. } => if_expr(
                eq(mcall(self.raw(x.clone()), "len", vec![], i64()), int(0)),
                text(&format!("array('{letter}')")),
                call(
                    &format!("zb_list_items_{suffix}"),
                    vec![
                        self.raw(x),
                        text(&format!("array('{letter}', [")),
                        text("])"),
                    ],
                    string(),
                ),
            ),
            Field::Tuple { suffix, .. } => {
                call(&format!("zb_tuple_repr_{suffix}"), vec![x], string())
            }
        }
    }

    fn boxed(&self, x: Expr) -> Expr {
        match self {
            Field::Int => call("zb_box_i64", vec![x], any()),
            Field::Float => call("zb_box_f64", vec![x], any()),
            Field::Bool => call("zb_box_bool", vec![x], any()),
            Field::Str => call("zb_box_str", vec![x], any()),
            Field::Any
            | Field::Dict { .. }
            | Field::Set { .. }
            | Field::List { .. }
            | Field::Array { .. } => x,
            Field::Instance { tag, .. } => call(
                "zb_box_instance",
                vec![cast(x, usize()), int32(*tag)],
                any(),
            ),
            Field::Tuple { suffix, .. } => call(&format!("zb_tuple_box_{suffix}"), vec![x], any()),
        }
    }

    /// The hash of a field value, as `zb_any_hash` hashes its boxed
    /// form, so a shaped tuple lands where its boxed twin does.
    fn hash(&self, x: Expr) -> Expr {
        match self {
            Field::Int => x,
            Field::Bool => cast(x, i64()),
            Field::Float => call("zb_hash_of_f64", vec![x], i64()),
            Field::Str => call("zb_hash_of_str", vec![x], i64()),
            Field::Tuple { suffix, .. } => call(&format!("zb_tuple_hash_{suffix}"), vec![x], i64()),
            Field::Any
            | Field::Dict { .. }
            | Field::Set { .. }
            | Field::List { .. }
            | Field::Array { .. } => call("zb_any_hash", vec![x], i64()),
            Field::Instance { .. } => call("zb_any_hash", vec![self.boxed(x)], i64()),
        }
    }

    /// Whether the box `stored` equals the field value `v`, as
    /// `zb_any_eq` would find its boxed form equal: a box of the
    /// field's own kind is read directly, anything else goes through
    /// the dynamic comparison.
    fn eq_boxed(&self, stored: Expr, v: Expr) -> Expr {
        let tag = |x: Expr| cast(call("zb_box_tag", vec![x], i32()), i64());
        let any_eq = |a: Expr, b: Expr| call("zb_any_eq", vec![a, b], boolean());
        match self {
            Field::Int => if_expr(
                eq(tag(stored.clone()), int(crate::dynamic::I64_TAG)),
                eq(
                    call("zb_box_payload_i64", vec![stored.clone()], i64()),
                    v.clone(),
                ),
                any_eq(stored, call("zb_box_i64", vec![v], any())),
            ),
            Field::Float => if_expr(
                eq(tag(stored.clone()), int(crate::dynamic::F64_TAG)),
                eq(
                    call("zb_box_payload_f64", vec![stored.clone()], f64()),
                    v.clone(),
                ),
                any_eq(stored, call("zb_box_f64", vec![v], any())),
            ),
            Field::Bool => any_eq(stored, call("zb_box_bool", vec![v], any())),
            Field::Str => and(
                eq(
                    call("zb_any_category", vec![stored.clone()], i64()),
                    int(crate::dynamic::STR),
                ),
                call(
                    "zb_str_eq",
                    vec![call("zb_box_get_str", vec![stored], string()), v],
                    boolean(),
                ),
            ),
            Field::Tuple { suffix, .. } => call(
                &format!("zb_tuple_eq_boxed_{suffix}"),
                vec![stored, v],
                boolean(),
            ),
            Field::Any
            | Field::Dict { .. }
            | Field::Set { .. }
            | Field::List { .. }
            | Field::Array { .. } => any_eq(stored, v),
            Field::Instance { .. } => any_eq(stored, self.boxed(v)),
        }
    }

    /// The stored form of a dynamic value: a dict, set or list stays the
    /// box it came in once checked, or a list of another kind is
    /// converted and boxed anew.
    fn read(&self, x: Expr) -> Expr {
        match self {
            Field::Int => call("zb_any_as_i64", vec![x], i64()),
            Field::Float => call("zb_any_as_f64", vec![x], f64()),
            Field::Bool => call("zb_any_as_bool", vec![x], boolean()),
            Field::Str => call("zb_any_as_str", vec![x], string()),
            Field::Any => x,
            Field::Dict { .. } => call("zb_dict_as_box", vec![x], any()),
            Field::Set { .. } => call("zb_set_as_box", vec![x], any()),
            Field::Instance { ty, tag } => cast(
                call("zb_hook_unbox_instance", vec![x, int32(*tag)], usize()),
                ty.clone(),
            ),
            Field::List { suffix, .. } => call(&format!("zb_list_as_box_{suffix}"), vec![x], any()),
            // The box itself once its tag is the array's; anything else
            // is the TypeError the check raises, and the value after it
            // is never read.
            Field::Array {
                suffix,
                ty,
                tag,
                letter,
            } => if_expr(
                eq(
                    cast(call("zb_box_tag", vec![x.clone()], i32()), i64()),
                    int(*tag),
                ),
                x.clone(),
                call(
                    &format!("zb_list_box_tagged_{suffix}"),
                    vec![
                        call(
                            &format!("zb_list_unbox_tagged_{suffix}"),
                            vec![x, int(*tag), text(letter)],
                            ty.clone(),
                        ),
                        int(*tag),
                    ],
                    any(),
                ),
            ),
            Field::Tuple { suffix, ty } => {
                call(&format!("zb_tuple_read_{suffix}"), vec![x], ty.clone())
            }
        }
    }
}

/// Kinds a frontend registers are numbered from here in a box's tag, so
/// none collides with the library's own kinds or a class's instances.
pub const SHAPE_KIND_BASE: i64 = 1 << 20;

/// The box tag of a list whose elements are the tuple shape `index`.
pub fn shape_list_tag(index: u16) -> i64 {
    ((SHAPE_KIND_BASE + index as i64) << 8) | 255
}

/// The functions of a tuple shape a frontend registers: equality,
/// order, repr, boxing and reading back of the tuple
/// (`zb_tuple_{eq,lt,repr,box,read}_<suffix>`). Shapes a field names
/// must have been declared before.
pub fn tuple_declarations(
    list_type: TypeId,
    suffix: &str,
    tuple_ty: Type,
    fields: &[Field],
) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let a = local("a", tuple_ty.clone());
    let b = local("b", tuple_ty.clone());
    let x = local("x", any());
    let t = local("t", anys.clone());
    let field = |v: &Local, i: usize| idx(v.e(), int(i as i64), fields[i].ty());
    let mut d = Vec::new();

    // Equal when every field is.
    let mut all = bool(true);
    for (i, f) in fields.iter().enumerate().rev() {
        let this = f.eq(field(&a, i), field(&b, i));
        all = if i + 1 == fields.len() {
            this
        } else {
            and(this, all)
        };
    }
    d.push(define(
        &format!("zb_tuple_eq_{suffix}"),
        &[&a, &b],
        boolean(),
        vec![ret(all)],
    ));
    // Ordered by the first field that differs.
    let mut order: Vec<Stmt> = Vec::new();
    for (i, f) in fields.iter().enumerate() {
        order.push(when(
            f.lt(field(&a, i), field(&b, i)),
            vec![ret(bool(true))],
        ));
        order.push(when(
            f.lt(field(&b, i), field(&a, i)),
            vec![ret(bool(false))],
        ));
    }
    order.push(ret(bool(false)));
    d.push(define(
        &format!("zb_tuple_lt_{suffix}"),
        &[&a, &b],
        boolean(),
        order,
    ));
    // `(x, y)`, with the comma a one-element tuple keeps.
    let mut text_of = text("(");
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            text_of = add(text_of, text(", "));
        }
        text_of = add(text_of, f.repr(field(&a, i)));
    }
    text_of = add(text_of, text(if fields.len() == 1 { ",)" } else { ")" }));
    d.push(define(
        &format!("zb_tuple_repr_{suffix}"),
        &[&a],
        string(),
        vec![ret(text_of)],
    ));
    // Boxed as the tagged list of the boxed fields.
    let boxed_fields: Vec<Expr> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| f.boxed(field(&a, i)))
        .collect();
    d.push(define(
        &format!("zb_tuple_box_{suffix}"),
        &[&a],
        any(),
        vec![ret(call(
            "zb_box_tuple",
            vec![list(boxed_fields, anys.clone())],
            any(),
        ))],
    ));
    // Read back once the tag and the length are checked.
    let read_fields: Vec<Expr> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| f.read(idx(t.e(), int(i as i64), any())))
        .collect();
    d.push(define(
        &format!("zb_tuple_read_{suffix}"),
        &[&x],
        tuple_ty.clone(),
        vec![
            t.decl(call("zb_unbox_tuple", vec![x.e()], anys.clone())),
            expr(call(
                "zb_list_expect_len_any",
                vec![t.e(), int(fields.len() as i64)],
                unit(),
            )),
            ret(tuple(read_fields, tuple_ty.clone())),
        ],
    ));
    // The hash its boxed form has, so a lookup by the value lands on
    // the box: the dynamic tuple hash's recurrence over the fields.
    let mut h = int(0x2545_F491_4F6C_DD1D);
    for (i, f) in fields.iter().enumerate() {
        h = add(mul(h, int(1_000_003)), f.hash(field(&a, i)));
    }
    d.push(define(
        &format!("zb_tuple_hash_{suffix}"),
        &[&a],
        i64(),
        vec![ret(h)],
    ));
    // Whether a box holds a tuple equal to the value: a tuple of the
    // arity whose every element equals the field.
    let mut same = vec![
        when(
            ne(
                cast(call("zb_box_tag", vec![x.e()], i32()), i64()),
                int(TUPLE_TAG),
            ),
            vec![ret(bool(false))],
        ),
        t.decl(call("zb_unbox_tuple", vec![x.e()], anys.clone())),
        when(
            ne(len(t.e()), int(fields.len() as i64)),
            vec![ret(bool(false))],
        ),
    ];
    for (i, f) in fields.iter().enumerate() {
        same.push(when(
            not(f.eq_boxed(idx(t.e(), int(i as i64), any()), field(&a, i))),
            vec![ret(bool(false))],
        ));
    }
    same.push(ret(bool(true)));
    d.push(define(
        &format!("zb_tuple_eq_boxed_{suffix}"),
        &[&x, &a],
        boolean(),
        same,
    ));
    // `value in set` and the dict lookups by the value, no box made.
    let hash_name = format!("zb_tuple_hash_{suffix}");
    let eq_name = format!("zb_tuple_eq_boxed_{suffix}");
    let box_name = format!("zb_tuple_box_{suffix}");
    let repr_name = format!("zb_tuple_repr_{suffix}");
    d.push(crate::dicts::set_contains_by(
        &format!("zb_set_contains_{suffix}"),
        list_type,
        tuple_ty.clone(),
        &|key| call(&hash_name, vec![key], i64()),
        &|stored, key| call(&eq_name, vec![stored, key], boolean()),
    ));
    d.extend(crate::dicts::dict_ops_by(
        list_type,
        suffix,
        tuple_ty.clone(),
        &|key| call(&hash_name, vec![key], i64()),
        &|stored, key| call(&eq_name, vec![stored, key], boolean()),
        &|key| call(&box_name, vec![key], any()),
        &|key| call(&repr_name, vec![key], string()),
    ));
    d
}

/// The list functions (`zb_list_*_<suffix>`) of a list whose elements
/// are the tuple shape `index`, over the shape's own functions from
/// [`tuple_declarations`].
pub fn tuple_list_declarations(
    list_type: TypeId,
    index: u16,
    suffix: &str,
    tuple_ty: Type,
) -> Vec<Decl> {
    let eq_name = format!("zb_tuple_eq_{suffix}");
    let lt_name = format!("zb_tuple_lt_{suffix}");
    let repr_name = format!("zb_tuple_repr_{suffix}");
    let box_name = format!("zb_tuple_box_{suffix}");
    let read_name = format!("zb_tuple_read_{suffix}");
    let elem = tuple_ty.clone();
    let k = KindOps {
        kind: None,
        suffix: suffix.to_string(),
        tag: shape_list_tag(index),
        list: list_of(list_type, elem.clone()),
        strs: list_of(list_type, string()),
        elem: elem.clone(),
        eq: Box::new(move |a, b| call(&eq_name, vec![a, b], boolean())),
        lt: Box::new(move |a, b| call(&lt_name, vec![a, b], boolean())),
        repr: Box::new(move |x| call(&repr_name, vec![x], string())),
        boxed: Box::new(move |x| call(&box_name, vec![x], any())),
        read: Box::new(move |x| call(&read_name, vec![x], elem.clone())),
    };
    kind_declarations(&k)
}

fn len(xs: Expr) -> Expr {
    mcall(xs, "len", vec![], i64())
}

/// A stable bottom-up merge sort of `xs`, ordered by `less` over the
/// elements, or over `keys` when given, in which case the keys move
/// with their elements. Runs are merged into a scratch copy and
/// written back after each pass; an element moves past another only
/// when `less` says so, never when the two compare equal. `copy` is the
/// element list's copy function.
fn merge_sort(
    xs: &Local,
    keys: Option<(&Local, &str)>,
    copy: &str,
    less: &dyn Fn(Expr, Expr) -> Expr,
) -> Vec<Stmt> {
    let elem_of = |ty: &Type| match ty {
        Type::Named { type_args, .. } if !type_args.is_empty() => type_args[0].clone(),
        other => other.clone(),
    };
    let elem = elem_of(&xs.ty);
    let key_elem = keys.map(|(ks, _)| elem_of(&ks.ty)).unwrap_or_else(any);
    // What `less` compares for the element at `i` of the source.
    let key_of = |i: Expr| match keys {
        Some((ks, _)) => idx(ks.e(), i, key_elem.clone()),
        None => idx(xs.e(), i, elem.clone()),
    };
    let n = local("n", i64());
    let tmp = local("tmp", xs.ty.clone());
    let ktmp = local(
        "ktmp",
        keys.map(|(ks, _)| ks.ty.clone()).unwrap_or_else(any),
    );
    let width = local("width", i64());
    let lo = local("lo", i64());
    let mid = local("mid", i64());
    let hi = local("hi", i64());
    let a = local("a", i64());
    let b = local("b", i64());
    let o = local("o", i64());
    let i = local("i", i64());

    // Move `src[from]` to `dst[o]`, keys alongside.
    let place = |from: &Local| {
        let mut s = vec![set_idx(tmp.e(), o.e(), idx(xs.e(), from.e(), elem.clone()))];
        if let Some((ks, _)) = keys {
            s.push(set_idx(
                ktmp.e(),
                o.e(),
                idx(ks.e(), from.e(), key_elem.clone()),
            ));
        }
        s.push(from.add_assign(int(1)));
        s
    };
    // The right run's element goes first only when it is strictly
    // less than the left's.
    let merge = vec![
        a.decl(lo.e()),
        b.decl(mid.e()),
        o.decl(lo.e()),
        while_(
            lt(o.e(), hi.e()),
            vec![
                if_(
                    ge(a.e(), mid.e()),
                    place(&b),
                    vec![if_(
                        ge(b.e(), hi.e()),
                        place(&a),
                        vec![if_(
                            less(key_of(b.e()), key_of(a.e())),
                            place(&b),
                            place(&a),
                        )],
                    )],
                ),
                o.add_assign(int(1)),
            ],
        ),
    ];
    let mut write_back = vec![set_idx(xs.e(), i.e(), idx(tmp.e(), i.e(), elem.clone()))];
    if let Some((ks, _)) = keys {
        write_back.push(set_idx(
            ks.e(),
            i.e(),
            idx(ktmp.e(), i.e(), key_elem.clone()),
        ));
    }
    let mut body = vec![n.decl(len(xs.e()))];
    body.push(when(lt(n.e(), int(2)), vec![ret_void()]));
    body.push(tmp.decl(call(copy, vec![xs.e()], xs.ty.clone())));
    if let Some((ks, key_copy)) = keys {
        body.push(ktmp.decl(call(key_copy, vec![ks.e()], ks.ty.clone())));
    }
    body.push(width.decl(int(1)));
    let mut pass = vec![lo.decl(int(0))];
    let mut one_merge = vec![
        mid.decl(add(lo.e(), width.e())),
        when(gt(mid.e(), n.e()), vec![mid.set(n.e())]),
        hi.decl(add(lo.e(), mul(width.e(), int(2)))),
        when(gt(hi.e(), n.e()), vec![hi.set(n.e())]),
    ];
    one_merge.extend(merge);
    one_merge.push(lo.set(add(lo.e(), mul(width.e(), int(2)))));
    pass.push(while_(lt(lo.e(), n.e()), one_merge));
    pass.extend(for_range(&i, int(0), n.e(), write_back));
    pass.push(width.set(mul(width.e(), int(2))));
    body.push(while_(lt(width.e(), n.e()), pass));
    body.push(ret_void());
    body
}

pub(crate) fn declarations(policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let mut out = vec![list_class()];
    for kind in Kind::LIBRARY {
        out.extend(kind_declarations(&ops(kind, list_type)));
    }
    out.extend(shared(policy, list_type));
    out
}

/// The list functions of an array storage kind, for a frontend to
/// generate into the program that uses arrays of it: the library holds
/// only its own kinds.
pub fn array_kind_declarations(kind: Kind, list_type: TypeId) -> Vec<Decl> {
    kind_declarations(&ops(kind, list_type))
}

fn kind_declarations(k: &KindOps) -> Vec<Decl> {
    let name = |op: &str| format!("zb_list_{op}_{}", k.suffix);
    // The list every operation works on is read or edited in place and
    // never kept, except by the box that carries it into a dynamic slot.
    let xs = borrowed("xs", k.list.clone());
    let carried = local("xs", k.list.clone());
    // The second list of a two-list operation is only read: extended
    // from, concatenated, compared, assigned from.
    let ys = borrowed("ys", k.list.clone());
    let out = local("out", k.list.clone());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());
    // The element a list keeps.
    let v = kept("v", k.elem.clone());
    let e = local("e", k.elem.clone());
    let what = local("what", string());
    let el = |xs: &Local, i: Expr| idx(xs.e(), i, k.elem.clone());
    let empty = || list(Vec::new(), k.list.clone());
    let mut d = Vec::new();
    if let Some(kind) = k.kind.filter(|kind| kind.wide() != *kind) {
        d.extend(narrow_declarations(k, kind));
    }

    // Index normalisation: negative counts from the end, out of range is
    // an IndexError.
    d.push(define(
        &name("norm"),
        &[&xs, &i, &what],
        i64(),
        vec![
            n.decl(len(xs.e())),
            j.decl(i.e()),
            when(lt(j.e(), int(0)), vec![j.set(add(j.e(), n.e()))]),
            when(
                or(lt(j.e(), int(0)), ge(j.e(), n.e())),
                vec![fatal("IndexError", what.e())],
            ),
            ret(j.e()),
        ],
    ));
    let norm = |i: Expr, msg: &str| call(&name("norm"), vec![xs.e(), i, text(msg)], i64());
    // Numeric storage checks its index inline: the check is what a hot
    // loop pays per element.
    let inline_check = k
        .kind
        .is_some_and(|kind| matches!(kind, Kind::Float | Kind::F32) || kind.is_narrow_int());
    let checked_index = |msg: &str| {
        if !inline_check {
            return vec![j.decl(norm(i.e(), msg))];
        }
        vec![
            n.decl(len(xs.e())),
            j.decl(i.e()),
            when(lt(j.e(), int(0)), vec![j.set(add(j.e(), n.e()))]),
            when(
                or(lt(j.e(), int(0)), ge(j.e(), n.e())),
                vec![fatal("IndexError", text(msg))],
            ),
        ]
    };
    let mut get_body = checked_index("list index out of range");
    get_body.push(ret(el(&xs, j.e())));
    d.push(define(&name("get"), &[&xs, &i], k.elem.clone(), get_body));
    // Used only after unpacking checked the sequence's exact length.
    d.push(define(
        &name("get_unchecked"),
        &[&xs, &i],
        k.elem.clone(),
        vec![ret(el(&xs, i.e()))],
    ));
    let mut set_body = checked_index("list assignment index out of range");
    set_body.push(set_idx(xs.e(), j.e(), v.e()));
    set_body.push(ret_void());
    d.push(define(&name("set"), &[&xs, &i, &v], unit(), set_body));
    d.push(define(
        &name("pop"),
        &[&xs, &i],
        k.elem.clone(),
        vec![
            when(
                eq(len(xs.e()), int(0)),
                vec![fatal("IndexError", text("pop from empty list"))],
            ),
            j.decl(norm(i.e(), "pop index out of range")),
            ret(mcall(xs.e(), "remove_at", vec![j.e()], k.elem.clone())),
        ],
    ));
    d.push(define(
        &name("insert"),
        &[&xs, &i, &v],
        unit(),
        vec![
            n.decl(len(xs.e())),
            j.decl(i.e()),
            when(
                lt(j.e(), int(0)),
                vec![
                    j.set(add(j.e(), n.e())),
                    when(lt(j.e(), int(0)), vec![j.set(int(0))]),
                ],
            ),
            when(gt(j.e(), n.e()), vec![j.set(n.e())]),
            expr(mcall(xs.e(), "insert_at", vec![j.e(), v.e()], unit())),
            ret_void(),
        ],
    ));
    d.push(define(&name("index"), &[&xs, &v], i64(), {
        let mut s = vec![n.decl(len(xs.e()))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                e.decl(el(&xs, i.e())),
                when((k.eq)(e.e(), v.e()), vec![ret(i.e())]),
            ],
        ));
        s.push(fatal(
            "ValueError",
            add((k.repr)(v.e()), text(" is not in list")),
        ));
        s.push(ret(int(0)));
        s
    }));
    d.push(define(&name("index_or_neg"), &[&xs, &v], i64(), {
        let mut s = vec![n.decl(len(xs.e()))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                e.decl(el(&xs, i.e())),
                when((k.eq)(e.e(), v.e()), vec![ret(i.e())]),
            ],
        ));
        s.push(ret(int(-1)));
        s
    }));
    d.push(define(&name("contains"), &[&xs, &v], boolean(), {
        let mut s = vec![n.decl(len(xs.e()))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                e.decl(el(&xs, i.e())),
                when((k.eq)(e.e(), v.e()), vec![ret(bool(true))]),
            ],
        ));
        s.push(ret(bool(false)));
        s
    }));
    let c = local("c", i64());
    d.push(define(&name("count"), &[&xs, &v], i64(), {
        let mut s = vec![n.decl(len(xs.e())), c.decl(int(0))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                e.decl(el(&xs, i.e())),
                when((k.eq)(e.e(), v.e()), vec![c.add_assign(int(1))]),
            ],
        ));
        s.push(ret(c.e()));
        s
    }));
    d.push(define(
        &name("remove"),
        &[&xs, &v],
        unit(),
        vec![
            i.decl(call(&name("index"), vec![xs.e(), v.e()], i64())),
            expr(mcall(xs.e(), "remove_at", vec![i.e()], k.elem.clone())),
            ret_void(),
        ],
    ));
    let a = local("a", k.elem.clone());
    let b = local("b", k.elem.clone());
    d.push(define(
        &name("reverse"),
        &[&xs],
        unit(),
        vec![
            i.decl(int(0)),
            j.decl(sub(len(xs.e()), int(1))),
            while_(
                lt(i.e(), j.e()),
                vec![
                    a.decl(el(&xs, i.e())),
                    b.decl(el(&xs, j.e())),
                    set_idx(xs.e(), i.e(), b.e()),
                    set_idx(xs.e(), j.e(), a.e()),
                    i.add_assign(int(1)),
                    j.set(sub(j.e(), int(1))),
                ],
            ),
            ret_void(),
        ],
    ));
    // Ascending order of the elements, stable.
    d.push(define(
        &name("sort"),
        &[&xs],
        unit(),
        merge_sort(&xs, None, &name("copy"), &|a, b| (k.lt)(a, b)),
    ));
    // One copy of the bytes: an element is the word or struct it is
    // stored as, whichever kind.
    d.push(define(
        &name("extend"),
        &[&xs, &ys],
        unit(),
        vec![
            expr(mcall(xs.e(), "append_all", vec![ys.e()], unit())),
            ret_void(),
        ],
    ));
    d.push(define(
        &name("copy"),
        &[&xs],
        k.list.clone(),
        vec![
            out.decl(empty()),
            expr(call(&name("extend"), vec![out.e(), xs.e()], unit())),
            ret(out.e()),
        ],
    ));
    d.push(define(
        &name("concat"),
        &[&xs, &ys],
        k.list.clone(),
        vec![
            out.decl(empty()),
            expr(call(&name("extend"), vec![out.e(), xs.e()], unit())),
            expr(call(&name("extend"), vec![out.e(), ys.e()], unit())),
            ret(out.e()),
        ],
    ));
    let times = local("times", i64());
    d.push(define(&name("repeat"), &[&xs, &times], k.list.clone(), {
        let mut s = vec![out.decl(empty())];
        s.extend(for_range(
            &i,
            int(0),
            times.e(),
            vec![expr(call(&name("extend"), vec![out.e(), xs.e()], unit()))],
        ));
        s.push(ret(out.e()));
        s
    }));
    let low = local("low", i64());
    let high = local("high", i64());
    let middle = local("middle", i64());
    for right in [false, true] {
        let search = if right { "bisect_right" } else { "bisect_left" };
        let goes_right = if right {
            not((k.lt)(v.e(), idx(xs.e(), middle.e(), k.elem.clone())))
        } else {
            (k.lt)(idx(xs.e(), middle.e(), k.elem.clone()), v.e())
        };
        d.push(define(
            &name(search),
            &[&xs, &v, &low, &high],
            i64(),
            vec![
                when(
                    lt(low.e(), int(0)),
                    vec![fatal("ValueError", text("lo must be non-negative"))],
                ),
                while_(
                    lt(low.e(), high.e()),
                    vec![
                        middle.decl(add(low.e(), div(sub(high.e(), low.e()), int(2)))),
                        if_(
                            goes_right,
                            vec![low.set(add(middle.e(), int(1)))],
                            vec![high.set(middle.e())],
                        ),
                    ],
                ),
                ret(low.e()),
            ],
        ));
        let insert = if right { "insort_right" } else { "insort_left" };
        d.push(define(
            &name(insert),
            &[&xs, &v, &low, &high],
            unit(),
            vec![
                i.decl(call(
                    &name(search),
                    vec![xs.e(), v.e(), low.e(), high.e()],
                    i64(),
                )),
                expr(mcall(xs.e(), "insert_at", vec![i.e(), v.e()], unit())),
                ret_void(),
            ],
        ));
    }
    // xs[start:stop:step]; `mask` bits 1, 2, 4 say which bounds were given.
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let mask = local("mask", i64());
    let st = local("st", i64());
    let lo = local("lo", i64());
    let hi = local("hi", i64());
    let step_body = |i: &Local| {
        vec![
            expr(mcall(out.e(), "push", vec![el(&xs, i.e())], unit())),
            i.set(add(i.e(), st.e())),
        ]
    };
    d.push(define(
        &name("slice"),
        &[&xs, &start, &stop, &step, &mask],
        k.list.clone(),
        vec![
            n.decl(len(xs.e())),
            st.decl(int(1)),
            when(ne(bitand(mask.e(), int(4)), int(0)), vec![st.set(step.e())]),
            when(
                eq(st.e(), int(0)),
                vec![fatal("ValueError", text("slice step cannot be zero"))],
            ),
            lo.decl(int(0)),
            hi.decl(n.e()),
            when(
                lt(st.e(), int(0)),
                vec![lo.set(sub(n.e(), int(1))), hi.set(int(-1))],
            ),
            when(
                ne(bitand(mask.e(), int(1)), int(0)),
                vec![lo.set(call(
                    "zb_slice_bound",
                    vec![start.e(), n.e(), st.e()],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(mask.e(), int(2)), int(0)),
                vec![hi.set(call("zb_slice_bound", vec![stop.e(), n.e(), st.e()], i64()))],
            ),
            out.decl(empty()),
            i.decl(lo.e()),
            if_(
                gt(st.e(), int(0)),
                vec![while_(lt(i.e(), hi.e()), step_body(&i))],
                vec![while_(gt(i.e(), hi.e()), step_body(&i))],
            ),
            ret(out.e()),
        ],
    ));
    // Replace selected elements in the same header. The source is read
    // as it is, unless it is the target's own storage, which is copied
    // first so the elements moved are the ones the slice named.
    let replacement = borrowed("replacement", k.list.clone());
    let selected = local("selected", i64());
    let width = local("width", i64());
    let common = local("common", i64());
    let data = |xs: &Local| fld(xs.e(), "data", i64());
    let assign_from = |source: Expr, step: Expr| {
        expr(call(
            &name("assign_slice_from"),
            vec![source, xs.e(), start.e(), stop.e(), step, mask.e()],
            unit(),
        ))
    };
    d.push(define(
        &name("assign_slice"),
        &[&ys, &xs, &start, &stop, &step, &mask],
        unit(),
        vec![
            when(
                eq(data(&ys), data(&xs)),
                vec![
                    replacement.decl(call(&name("copy"), vec![ys.e()], k.list.clone())),
                    assign_from(replacement.e(), step.e()),
                    ret_void(),
                ],
            ),
            assign_from(ys.e(), step.e()),
            ret_void(),
        ],
    ));
    // xs[start:stop] = xs[rstart:rstop:-1]. The two ranges naming the
    // same elements is a reversal in place; anything else is the slice
    // taken first and assigned as any other source.
    let rstart = local("rstart", i64());
    let rstop = local("rstop", i64());
    let rmask = local("rmask", i64());
    let rlo = local("rlo", i64());
    let rhi = local("rhi", i64());
    let taken = local("taken", k.list.clone());
    d.push(define(
        &name("assign_reversed_slice"),
        &[&xs, &start, &stop, &mask, &rstart, &rstop, &rmask],
        unit(),
        vec![
            n.decl(len(xs.e())),
            lo.decl(int(0)),
            hi.decl(n.e()),
            when(
                ne(bitand(mask.e(), int(1)), int(0)),
                vec![lo.set(call(
                    "zb_slice_bound",
                    vec![start.e(), n.e(), int(1)],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(mask.e(), int(2)), int(0)),
                vec![hi.set(call("zb_slice_bound", vec![stop.e(), n.e(), int(1)], i64()))],
            ),
            when(lt(hi.e(), lo.e()), vec![hi.set(lo.e())]),
            rlo.decl(sub(n.e(), int(1))),
            rhi.decl(int(-1)),
            when(
                ne(bitand(rmask.e(), int(1)), int(0)),
                vec![rlo.set(call(
                    "zb_slice_bound",
                    vec![rstart.e(), n.e(), int(-1)],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(rmask.e(), int(2)), int(0)),
                vec![rhi.set(call(
                    "zb_slice_bound",
                    vec![rstop.e(), n.e(), int(-1)],
                    i64(),
                ))],
            ),
            if_(
                and(
                    eq(rlo.e(), sub(hi.e(), int(1))),
                    eq(rhi.e(), sub(lo.e(), int(1))),
                ),
                vec![
                    i.decl(lo.e()),
                    j.decl(sub(hi.e(), int(1))),
                    while_(
                        lt(i.e(), j.e()),
                        vec![
                            a.decl(el(&xs, i.e())),
                            b.decl(el(&xs, j.e())),
                            set_idx(xs.e(), i.e(), b.e()),
                            set_idx(xs.e(), j.e(), a.e()),
                            i.add_assign(int(1)),
                            j.set(sub(j.e(), int(1))),
                        ],
                    ),
                ],
                vec![
                    taken.decl(call(
                        &name("slice"),
                        vec![
                            xs.e(),
                            rstart.e(),
                            rstop.e(),
                            int(-1),
                            bitor(rmask.e(), int(4)),
                        ],
                        k.list.clone(),
                    )),
                    assign_from(taken.e(), int(0)),
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(
        &name("assign_slice_from"),
        &[&replacement, &xs, &start, &stop, &step, &mask],
        unit(),
        vec![
            n.decl(len(xs.e())),
            st.decl(int(1)),
            when(ne(bitand(mask.e(), int(4)), int(0)), vec![st.set(step.e())]),
            when(
                eq(st.e(), int(0)),
                vec![fatal("ValueError", text("slice step cannot be zero"))],
            ),
            lo.decl(int(0)),
            hi.decl(n.e()),
            when(
                lt(st.e(), int(0)),
                vec![lo.set(sub(n.e(), int(1))), hi.set(int(-1))],
            ),
            when(
                ne(bitand(mask.e(), int(1)), int(0)),
                vec![lo.set(call(
                    "zb_slice_bound",
                    vec![start.e(), n.e(), st.e()],
                    i64(),
                ))],
            ),
            when(
                ne(bitand(mask.e(), int(2)), int(0)),
                vec![hi.set(call("zb_slice_bound", vec![stop.e(), n.e(), st.e()], i64()))],
            ),
            selected.decl(len(replacement.e())),
            if_(
                ne(st.e(), int(1)),
                vec![
                    width.decl(int(0)),
                    i.decl(lo.e()),
                    if_(
                        gt(st.e(), int(0)),
                        vec![while_(
                            lt(i.e(), hi.e()),
                            vec![width.add_assign(int(1)), i.set(add(i.e(), st.e()))],
                        )],
                        vec![while_(
                            gt(i.e(), hi.e()),
                            vec![width.add_assign(int(1)), i.set(add(i.e(), st.e()))],
                        )],
                    ),
                    when(
                        ne(width.e(), selected.e()),
                        vec![fatal(
                            "ValueError",
                            text("attempt to assign sequence of wrong size to extended slice"),
                        )],
                    ),
                    i.decl(lo.e()),
                    j.decl(int(0)),
                    while_(
                        lt(j.e(), selected.e()),
                        vec![
                            set_idx(xs.e(), i.e(), el(&replacement, j.e())),
                            i.set(add(i.e(), st.e())),
                            j.add_assign(int(1)),
                        ],
                    ),
                    ret_void(),
                ],
                Vec::new(),
            ),
            when(lt(hi.e(), lo.e()), vec![hi.set(lo.e())]),
            width.decl(sub(hi.e(), lo.e())),
            common.decl(width.e()),
            when(gt(common.e(), selected.e()), vec![common.set(selected.e())]),
            i.decl(int(0)),
            while_(
                lt(i.e(), common.e()),
                vec![
                    set_idx(xs.e(), add(lo.e(), i.e()), el(&replacement, i.e())),
                    i.add_assign(int(1)),
                ],
            ),
            if_(
                lt(selected.e(), width.e()),
                vec![
                    j.decl(selected.e()),
                    while_(
                        lt(j.e(), width.e()),
                        vec![
                            expr(mcall(
                                xs.e(),
                                "remove_at",
                                vec![add(lo.e(), selected.e())],
                                k.elem.clone(),
                            )),
                            j.add_assign(int(1)),
                        ],
                    ),
                ],
                vec![
                    j.decl(width.e()),
                    while_(
                        lt(j.e(), selected.e()),
                        vec![
                            expr(mcall(
                                xs.e(),
                                "insert_at",
                                vec![add(lo.e(), j.e()), el(&replacement, j.e())],
                                unit(),
                            )),
                            j.add_assign(int(1)),
                        ],
                    ),
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(&name("eq"), &[&xs, &ys], boolean(), {
        let mut s = vec![
            n.decl(len(xs.e())),
            when(ne(n.e(), len(ys.e())), vec![ret(bool(false))]),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                a.decl(el(&xs, i.e())),
                b.decl(el(&ys, i.e())),
                when(not((k.eq)(a.e(), b.e())), vec![ret(bool(false))]),
            ],
        ));
        s.push(ret(bool(true)));
        s
    }));
    let m = local("m", i64());
    d.push(define(
        &name("lt"),
        &[&xs, &ys],
        boolean(),
        vec![
            n.decl(len(xs.e())),
            m.decl(len(ys.e())),
            i.decl(int(0)),
            while_(
                and(lt(i.e(), n.e()), lt(i.e(), m.e())),
                vec![
                    a.decl(el(&xs, i.e())),
                    b.decl(el(&ys, i.e())),
                    when((k.lt)(a.e(), b.e()), vec![ret(bool(true))]),
                    when((k.lt)(b.e(), a.e()), vec![ret(bool(false))]),
                    i.add_assign(int(1)),
                ],
            ),
            ret(lt(n.e(), m.e())),
        ],
    ));
    // The text is gathered as pieces and joined once, so a long list
    // prints in time and memory proportional to its text.
    let open = local("open", string());
    let close = local("close", string());
    let pieces = local("pieces", k.strs.clone());
    let piece = |p: Expr| expr(mcall(pieces.e(), "push", vec![p], unit()));
    d.push(define(&name("items"), &[&xs, &open, &close], string(), {
        let mut s = vec![
            pieces.decl(list(Vec::new(), k.strs.clone())),
            piece(open.e()),
            n.decl(len(xs.e())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![
                when(gt(i.e(), int(0)), vec![piece(text(", "))]),
                e.decl(el(&xs, i.e())),
                piece((k.repr)(e.e())),
            ],
        ));
        s.push(piece(close.e()));
        s.push(ret(call(
            "zb_str_join",
            vec![text(""), pieces.e()],
            string(),
        )));
        s
    }));
    d.push(define(
        &name("repr"),
        &[&xs],
        string(),
        vec![ret(call(
            &name("items"),
            vec![xs.e(), text("["), text("]")],
            string(),
        ))],
    ));
    let best = local("best", k.elem.clone());
    for (op, message, better) in [
        ("min", "min() arg is an empty sequence", true),
        ("max", "max() arg is an empty sequence", false),
    ] {
        let pick: Expr = if better {
            (k.lt)(e.e(), best.e())
        } else {
            (k.lt)(best.e(), e.e())
        };
        let mut s = vec![
            when(
                eq(len(xs.e()), int(0)),
                vec![fatal("ValueError", text(message))],
            ),
            best.decl(el(&xs, int(0))),
            n.decl(len(xs.e())),
        ];
        s.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![e.decl(el(&xs, i.e())), when(pick, vec![best.set(e.e())])],
        ));
        s.push(ret(best.e()));
        d.push(define(&name(op), &[&xs], k.elem.clone(), s));
    }
    // Ordering by keys computed elsewhere: `keys[i]` is the key of
    // `xs[i]`, one dynamic value per element. Both lists move together,
    // so the sort stays stable in either direction: an element moves past
    // another only when its key is strictly smaller (or, descending,
    // strictly greater), never when the two are equal.
    let any_list = list_of(list_type_of(&k.list), any());
    let keys = local("keys", any_list.clone());
    let descending = local("descending", boolean());
    let key = local("key", any());
    // One sort per direction, chosen once: a direction test inside
    // the comparison would be paid at every step. Keys that are all
    // ints, all floats or all strings are read out once and compared
    // as such, since the dynamic comparison would settle the same
    // question at every step.
    let sort_with = |ks: &Local, key_copy: &str, less: &dyn Fn(Expr, Expr) -> Expr| {
        vec![if_(
            descending.e(),
            merge_sort(&xs, Some((ks, key_copy)), &name("copy"), &|a, b| less(b, a)),
            merge_sort(&xs, Some((ks, key_copy)), &name("copy"), less),
        )]
    };
    let ikeys = local("ikeys", list_of(list_type_of(&k.list), i64()));
    let fkeys = local("fkeys", list_of(list_type_of(&k.list), f64()));
    let skeys = local("skeys", list_of(list_type_of(&k.list), string()));
    d.push(define(
        &name("sort_by_int_keys"),
        &[&xs, &ikeys, &descending],
        unit(),
        sort_with(&ikeys, "zb_list_copy_i64", &|a, b| lt(a, b)),
    ));
    d.push(define(
        &name("sort_by_float_keys"),
        &[&xs, &fkeys, &descending],
        unit(),
        sort_with(&fkeys, "zb_list_copy_f64", &|a, b| lt(a, b)),
    ));
    d.push(define(
        &name("sort_by_str_keys"),
        &[&xs, &skeys, &descending],
        unit(),
        sort_with(&skeys, "zb_list_copy_str", &|a, b| {
            call("zb_str_lt", vec![a, b], boolean())
        }),
    ));
    d.push(define(
        &name("sort_by_any_keys"),
        &[&xs, &keys, &descending],
        unit(),
        sort_with(&keys, "zb_list_copy_any", &|a, b| {
            call("zb_any_lt", vec![a, b], boolean())
        }),
    ));
    let key_kind = local("kind", i64());
    let typed_keys = |op: &str, from: &str, ks: &Local| {
        vec![
            ks.decl(call(from, vec![keys.e()], ks.ty.clone())),
            expr(call(
                &name(op),
                vec![xs.e(), ks.e(), descending.e()],
                unit(),
            )),
            ret_void(),
        ]
    };
    d.push(define(
        &name("sort_by"),
        &[&xs, &keys, &descending],
        unit(),
        {
            let mut s = vec![
                n.decl(len(keys.e())),
                when(lt(n.e(), int(2)), vec![ret_void()]),
                key_kind.decl(call(
                    "zb_any_key_kind",
                    vec![idx(keys.e(), int(0), any())],
                    i64(),
                )),
                i.decl(int(1)),
                while_(
                    and(lt(i.e(), n.e()), ne(key_kind.e(), int(0))),
                    vec![
                        when(
                            ne(
                                call("zb_any_key_kind", vec![idx(keys.e(), i.e(), any())], i64()),
                                key_kind.e(),
                            ),
                            vec![key_kind.set(int(0))],
                        ),
                        i.add_assign(int(1)),
                    ],
                ),
                when(
                    eq(key_kind.e(), int(1)),
                    typed_keys("sort_by_int_keys", "zb_list_from_any_i64", &ikeys),
                ),
                when(
                    eq(key_kind.e(), int(2)),
                    typed_keys("sort_by_float_keys", "zb_list_from_any_f64", &fkeys),
                ),
                when(
                    eq(key_kind.e(), int(3)),
                    typed_keys("sort_by_str_keys", "zb_list_from_any_str", &skeys),
                ),
                expr(call(
                    &name("sort_by_any_keys"),
                    vec![xs.e(), keys.e(), descending.e()],
                    unit(),
                )),
                ret_void(),
            ];
            s.shrink_to_fit();
            s
        },
    ));
    // Descending order of the elements themselves, stable like `sort`.
    d.push(define(
        &name("sort_desc"),
        &[&xs],
        unit(),
        merge_sort(&xs, None, &name("copy"), &|a, b| (k.lt)(b, a)),
    ));
    // The element whose key is least (or greatest); the first of equals.
    let best_key = local("best_key", any());
    for (op, message, better) in [
        ("min_by", "min() arg is an empty sequence", true),
        ("max_by", "max() arg is an empty sequence", false),
    ] {
        let pick: Expr = if better {
            call("zb_any_lt", vec![key.e(), best_key.e()], boolean())
        } else {
            call("zb_any_lt", vec![best_key.e(), key.e()], boolean())
        };
        let mut s = vec![
            when(
                eq(len(xs.e()), int(0)),
                vec![fatal("ValueError", text(message))],
            ),
            best.decl(el(&xs, int(0))),
            best_key.decl(idx(keys.e(), int(0), any())),
            n.decl(len(xs.e())),
        ];
        s.extend(for_range(
            &i,
            int(1),
            n.e(),
            vec![
                key.decl(idx(keys.e(), i.e(), any())),
                when(pick, vec![best.set(el(&xs, i.e())), best_key.set(key.e())]),
            ],
        ));
        s.push(ret(best.e()));
        d.push(define(&name(op), &[&xs, &keys], k.elem.clone(), s));
    }
    // Everything boxed, for a list that becomes dynamic.
    let out_any = local("out", any_list.clone());
    let boxed = |e: Expr| (k.boxed)(e);
    d.push(define(&name("to_any"), &[&xs], any_list.clone(), {
        let mut s = vec![
            out_any.decl(list(Vec::new(), any_list.clone())),
            n.decl(len(xs.e())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(
                out_any.e(),
                "push",
                vec![boxed(el(&xs, i.e()))],
                unit(),
            ))],
        ));
        s.push(ret(out_any.e()));
        s
    }));
    // The reverse: each element read back as this kind, or a TypeError.
    // An instance list takes the class tag its elements must carry.
    let anys_in = local("xs", any_list.clone());
    let tag = local("tag", i32());
    let out_typed = local("out", k.list.clone());
    let read = |e: Expr| (k.read)(e);
    let from_params: Vec<&Local> = match k.kind {
        Some(Kind::Ptr) => vec![&anys_in, &tag],
        _ => vec![&anys_in],
    };
    d.push(define(&name("from_any"), &from_params, k.list.clone(), {
        let mut s = vec![
            out_typed.decl(list(Vec::new(), k.list.clone())),
            n.decl(len(anys_in.e())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(
                out_typed.e(),
                "push",
                vec![read(idx(anys_in.e(), i.e(), any()))],
                unit(),
            ))],
        ));
        s.push(ret(out_typed.e()));
        s
    }));
    // Unpacking: exactly `n` elements.
    let have = local("have", i64());
    d.push(define(
        &name("expect_len"),
        &[&xs, &n],
        unit(),
        vec![
            have.decl(len(xs.e())),
            when(
                lt(have.e(), n.e()),
                vec![fatal(
                    "ValueError",
                    add(
                        add(
                            add(
                                text("not enough values to unpack (expected "),
                                call("zb_str_of_int", vec![n.e()], string()),
                            ),
                            text(", got "),
                        ),
                        add(call("zb_str_of_int", vec![have.e()], string()), text(")")),
                    ),
                )],
            ),
            when(
                gt(have.e(), n.e()),
                vec![fatal(
                    "ValueError",
                    add(
                        add(
                            text("too many values to unpack (expected "),
                            call("zb_str_of_int", vec![n.e()], string()),
                        ),
                        text(")"),
                    ),
                )],
            ),
            ret_void(),
        ],
    ));
    // A list flows into a dynamic slot by reference, under the tag of
    // its kind, and comes back out by checking that tag.
    d.push(extern_fn(
        &format!("zb_box_list_raw_{}", k.suffix),
        &[("xs", k.list.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    d.push(extern_fn(
        &format!("zb_unbox_list_raw_{}", k.suffix),
        &[("x", any())],
        k.list.clone(),
        Some("zyntax_box_pointer"),
    ));
    d.push(define(
        &name("box"),
        &[&carried],
        any(),
        vec![ret(call(
            &format!("zb_box_list_raw_{}", k.suffix),
            vec![carried.e(), int32(k.tag as i32)],
            any(),
        ))],
    ));
    // The same box under a tag the caller chooses: an array's, which
    // carries its typecode.
    let chosen = local("tag", i64());
    d.push(define(
        &name("box_tagged"),
        &[&carried, &chosen],
        any(),
        vec![ret(call(
            &format!("zb_box_list_raw_{}", k.suffix),
            vec![carried.e(), cast(chosen.e(), i32())],
            any(),
        ))],
    ));
    let x = local("x", any());
    // A list of dynamic values is read out of a box of any list kind,
    // and a list of a primitive kind out of a box of dynamic values
    // whose every element is one; either comes out as a converted copy.
    let tag = local("tag", i64());
    let mut unbox = vec![tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64()))];
    match k.kind {
        Some(Kind::Any) => {
            for other in Kind::LIBRARY.iter().filter(|o| **o != Kind::Any) {
                unbox.push(when(
                    eq(tag.e(), int(other.list_tag())),
                    vec![ret(call(
                        &format!("zb_list_to_any_{}", other.suffix()),
                        vec![call(
                            &format!("zb_unbox_list_raw_{}", other.suffix()),
                            vec![x.e()],
                            list_of(list_type_of(&k.list), other.ty()),
                        )],
                        k.list.clone(),
                    ))],
                ));
            }
            // A list of a shape a frontend registered comes out through
            // its hook, as the frontend's kinds do.
            unbox.push(when(
                ge(tag.e(), int(SHAPE_KIND_BASE << 8)),
                vec![ret(call(
                    "zb_hook_shaped_items",
                    vec![x.e()],
                    k.list.clone(),
                ))],
            ));
        }
        Some(Kind::Ptr) => {}
        _ => {
            unbox.push(when(
                eq(tag.e(), int(Kind::Any.list_tag())),
                vec![ret(call(
                    &name("from_any"),
                    vec![call(
                        "zb_unbox_list_raw_any",
                        vec![x.e()],
                        list_of(list_type_of(&k.list), any()),
                    )],
                    k.list.clone(),
                ))],
            ));
        }
    }
    unbox.push(when(
        ne(tag.e(), int(k.tag)),
        vec![fatal(
            "TypeError",
            add(
                text("expected a list, got "),
                call("zb_any_type", vec![x.e()], string()),
            ),
        )],
    ));
    unbox.push(ret(call(
        &format!("zb_unbox_list_raw_{}", k.suffix),
        vec![x.e()],
        k.list.clone(),
    )));
    d.push(define(&name("unbox"), &[&x], k.list.clone(), unbox));
    // A box read back as the array whose tag is `tag`: a list stored
    // the same way, or an array of another typecode, is a TypeError.
    let wanted = local("tag", i64());
    let letter = local("letter", string());
    d.push(define(
        &name("unbox_tagged"),
        &[&x, &wanted, &letter],
        k.list.clone(),
        vec![
            when(
                ne(
                    cast(call("zb_box_tag", vec![x.e()], i32()), i64()),
                    wanted.e(),
                ),
                vec![fatal(
                    "TypeError",
                    add(
                        add(add(text("expected array('"), letter.e()), text("'), got ")),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call(
                &format!("zb_unbox_list_raw_{}", k.suffix),
                vec![x.e()],
                k.list.clone(),
            )),
        ],
    ));
    // A box holding a list of this kind, from a dynamic value: the box
    // itself when its tag is the kind's, else the checked conversion
    // boxed anew.
    d.push(define(
        &name("as_box"),
        &[&x],
        any(),
        vec![
            when(
                eq(
                    cast(call("zb_box_tag", vec![x.e()], i32()), i64()),
                    int(k.tag),
                ),
                vec![ret(x.e())],
            ),
            ret(call(
                &name("box"),
                vec![call(&name("unbox"), vec![x.e()], k.list.clone())],
                any(),
            )),
        ],
    ));
    d
}

/// What array storage adds to a kind: the number an element reads as
/// is wider than the element, so a value is checked on the way in and
/// widened on the way out.
///
/// `zb_list_narrow_<k>(v)` is the element a number becomes, or an
/// OverflowError worded as CPython words it (a `b` value beyond a short
/// is reported as a short, an `H` value beyond an int as an int, since
/// CPython converts in those steps). `zb_list_in_range_<k>(v)` is
/// whether a number could be an element at all, which a search asks
/// before narrowing. `zb_list_from_wide_<k>` and `zb_list_to_wide_<k>`
/// convert whole lists, and `zb_list_sum_<k>` adds up in the wide type.
fn narrow_declarations(k: &KindOps, kind: Kind) -> Vec<Decl> {
    let name = |op: &str| format!("zb_list_{op}_{}", k.suffix);
    let wide = kind.wide().ty();
    let wide_list = list_of(list_type_of(&k.list), wide.clone());
    let v = local("v", wide.clone());
    let i = local("i", i64());
    let n = local("n", i64());
    let mut d = Vec::new();

    // (below, message) and (above, message) bounds, checked in order.
    let bounds: Vec<(Expr, &str)> = match kind {
        Kind::I8 => vec![
            (
                lt(v.e(), int(-32768)),
                "signed short integer is less than minimum",
            ),
            (
                gt(v.e(), int(32767)),
                "signed short integer is greater than maximum",
            ),
            (lt(v.e(), int(-128)), "signed char is less than minimum"),
            (gt(v.e(), int(127)), "signed char is greater than maximum"),
        ],
        Kind::U8 => vec![
            (
                lt(v.e(), int(0)),
                "unsigned byte integer is less than minimum",
            ),
            (
                gt(v.e(), int(255)),
                "unsigned byte integer is greater than maximum",
            ),
        ],
        Kind::I16 => vec![
            (
                lt(v.e(), int(-32768)),
                "signed short integer is less than minimum",
            ),
            (
                gt(v.e(), int(32767)),
                "signed short integer is greater than maximum",
            ),
        ],
        Kind::U16 => vec![
            (
                lt(v.e(), int(-2147483648)),
                "signed integer is less than minimum",
            ),
            (
                gt(v.e(), int(2147483647)),
                "signed integer is greater than maximum",
            ),
            (lt(v.e(), int(0)), "unsigned short is less than minimum"),
            (
                gt(v.e(), int(65535)),
                "unsigned short is greater than maximum",
            ),
        ],
        Kind::I32 => vec![
            (
                lt(v.e(), int(-2147483648)),
                "signed integer is less than minimum",
            ),
            (
                gt(v.e(), int(2147483647)),
                "signed integer is greater than maximum",
            ),
        ],
        Kind::U32 => vec![
            (
                lt(v.e(), int(0)),
                "can't convert negative value to unsigned int",
            ),
            (
                gt(v.e(), int(4294967295)),
                "unsigned int is greater than maximum",
            ),
        ],
        Kind::U64 => vec![(
            lt(v.e(), int(0)),
            "can't convert negative value to unsigned int",
        )],
        // A float narrows by rounding.
        _ => Vec::new(),
    };
    let checks: Vec<Stmt> = bounds
        .iter()
        .map(|(out_of_range, message)| {
            when(
                out_of_range.clone(),
                vec![fatal("OverflowError", text(message))],
            )
        })
        .collect();
    let mut narrow = checks.clone();
    narrow.push(ret(cast(v.e(), k.elem.clone())));
    d.push(define(&name("narrow"), &[&v], k.elem.clone(), narrow));
    // A number no element can equal: out of range, or a float the
    // storage cannot hold exactly.
    let fits: Expr = match kind {
        Kind::F32 => eq(cast(cast(v.e(), k.elem.clone()), wide.clone()), v.e()),
        _ => bounds
            .iter()
            .map(|(out_of_range, _)| not(out_of_range.clone()))
            .reduce(and)
            .unwrap_or_else(|| bool(true)),
    };
    d.push(define(&name("in_range"), &[&v], boolean(), vec![ret(fits)]));

    let ws = borrowed("ws", wide_list.clone());
    let out = local("out", k.list.clone());
    d.push(define(&name("from_wide"), &[&ws], k.list.clone(), {
        let mut s = vec![
            n.decl(len(ws.e())),
            out.decl(list(Vec::new(), k.list.clone())),
            expr(mcall(out.e(), "reserve", vec![n.e()], unit())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(
                out.e(),
                "push",
                vec![call(
                    &name("narrow"),
                    vec![idx(ws.e(), i.e(), wide.clone())],
                    k.elem.clone(),
                )],
                unit(),
            ))],
        ));
        s.push(ret(out.e()));
        s
    }));
    let xs = borrowed("xs", k.list.clone());
    let wout = local("out", wide_list.clone());
    d.push(define(&name("to_wide"), &[&xs], wide_list.clone(), {
        let mut s = vec![
            n.decl(len(xs.e())),
            wout.decl(list(Vec::new(), wide_list.clone())),
            expr(mcall(wout.e(), "reserve", vec![n.e()], unit())),
        ];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(
                wout.e(),
                "push",
                vec![cast(idx(xs.e(), i.e(), k.elem.clone()), wide.clone())],
                unit(),
            ))],
        ));
        s.push(ret(wout.e()));
        s
    }));
    let total = local("s", wide.clone());
    let zero = match kind {
        Kind::F32 => float(0.0),
        _ => int(0),
    };
    d.push(define(&name("sum"), &[&xs], wide.clone(), {
        let mut s = vec![total.decl(zero), n.decl(len(xs.e()))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![total.set(add(
                total.e(),
                cast(idx(xs.e(), i.e(), k.elem.clone()), wide.clone()),
            ))],
        ));
        s.push(ret(total.e()));
        s
    }));
    d
}

fn list_type_of(list: &Type) -> TypeId {
    match list {
        Type::Named { id, .. } => *id,
        _ => unreachable!("a list type is Named"),
    }
}

/// What is not per kind: sums, ranges, tuples, strings as lists.
/// What instance addresses in a list need beyond the kind's own
/// functions: the hook that boxes one, which a frontend with classes
/// defines and which otherwise boxes the address as an opaque value,
/// and the refusal to order two of them.
pub(crate) fn ptr_declarations(policy: &Policy) -> Vec<Decl> {
    let a = local("a", usize());
    let b = local("b", usize());
    let p = local("p", usize());
    let x = local("x", any());
    let tag = local("tag", i32());
    let mut d = Vec::new();
    if policy.instance_hooks {
        d.push(extern_fn(
            "zb_hook_box_instance",
            &[("p", usize())],
            any(),
            None,
        ));
        // The address in a box carrying `tag` (or a tag the frontend
        // takes for one of its kind), or a TypeError.
        d.push(extern_fn(
            "zb_hook_unbox_instance",
            &[("x", any()), ("tag", i32())],
            usize(),
            None,
        ));
    } else {
        d.push(define(
            "zb_hook_box_instance",
            &[&p],
            any(),
            vec![ret(call(
                "zb_box_fnptr_raw",
                vec![p.e(), int32(255)],
                any(),
            ))],
        ));
        d.push(define(
            "zb_hook_unbox_instance",
            &[&x, &tag],
            usize(),
            vec![
                when(
                    ne(call("zb_box_tag", vec![x.e()], i32()), tag.e()),
                    vec![fatal("TypeError", text("not an address of that kind"))],
                ),
                ret(cast(
                    call("zb_unbox_instance_raw", vec![x.e()], i64()),
                    usize(),
                )),
            ],
        ));
    }
    d.push(define(
        "zb_ptr_lt",
        &[&a, &b],
        boolean(),
        vec![
            fatal(
                "TypeError",
                text("'<' not supported between instances of these objects"),
            ),
            ret(bool(false)),
        ],
    ));
    d
}

/// What the dynamic layer asks of a boxed list whose elements are a
/// shape a frontend registered (a tag from [`SHAPE_KIND_BASE`] up): the
/// frontend, which knows the shapes, defines these; a language without
/// shapes gets versions that report the kind unknown.
pub(crate) fn shape_hook_declarations(policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let anys = list_of(list_type, any());
    let x = local("x", any());
    let i = local("i", i64());
    let v = kept("v", any());
    let mut d = Vec::new();
    if policy.instance_hooks {
        d.push(extern_fn(
            "zb_hook_shaped_items",
            &[("x", any())],
            anys.clone(),
            None,
        ));
        d.push(extern_fn(
            "zb_hook_shaped_get",
            &[("x", any()), ("i", i64())],
            any(),
            None,
        ));
        d.push(extern_fn(
            "zb_hook_shaped_set",
            &[("x", any()), ("i", i64()), ("v", any())],
            unit(),
            None,
        ));
        d.push(extern_fn(
            "zb_hook_shaped_append",
            &[("x", any()), ("v", any())],
            unit(),
            None,
        ));
        d.push(extern_fn(
            "zb_hook_shaped_repr",
            &[("x", any())],
            string(),
            None,
        ));
        d.push(extern_fn(
            "zb_hook_shaped_assign_slice",
            &[
                ("x", any()),
                ("ys", anys.clone()),
                ("start", i64()),
                ("stop", i64()),
                ("step", i64()),
                ("mask", i64()),
            ],
            unit(),
            None,
        ));
        return d;
    }
    let unknown = || fatal("TypeError", text("a list of an unknown kind"));
    d.push(define(
        "zb_hook_shaped_items",
        &[&x],
        anys.clone(),
        vec![unknown(), ret(list(Vec::new(), anys.clone()))],
    ));
    d.push(define(
        "zb_hook_shaped_get",
        &[&x, &i],
        any(),
        vec![unknown(), ret(null(any()))],
    ));
    d.push(define(
        "zb_hook_shaped_set",
        &[&x, &i, &v],
        unit(),
        vec![unknown(), ret_void()],
    ));
    d.push(define(
        "zb_hook_shaped_append",
        &[&x, &v],
        unit(),
        vec![unknown(), ret_void()],
    ));
    d.push(define(
        "zb_hook_shaped_repr",
        &[&x],
        string(),
        vec![unknown(), ret(text(""))],
    ));
    let ys = borrowed("ys", anys.clone());
    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let mask = local("mask", i64());
    d.push(define(
        "zb_hook_shaped_assign_slice",
        &[&x, &ys, &start, &stop, &step, &mask],
        unit(),
        vec![unknown(), ret_void()],
    ));
    d
}

fn shared(_policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let ints = list_of(list_type, i64());
    let floats = list_of(list_type, f64());
    let strs = list_of(list_type, string());
    let anys = list_of(list_type, any());
    let i = local("i", i64());
    let n = local("n", i64());
    let mut d = Vec::new();

    let xs = local("xs", ints.clone());
    let s = local("s", i64());
    d.push(define("zb_list_sum_i64", &[&xs], i64(), {
        let mut st = vec![s.decl(int(0)), n.decl(len(xs.e()))];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![s.set(add(s.e(), idx(xs.e(), i.e(), i64())))],
        ));
        st.push(ret(s.e()));
        st
    }));
    let xf = local("xs", floats.clone());
    let sf = local("s", f64());
    d.push(define("zb_list_sum_f64", &[&xf], f64(), {
        let mut st = vec![sf.decl(float(0.0)), n.decl(len(xf.e()))];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![sf.set(add(sf.e(), idx(xf.e(), i.e(), f64())))],
        ));
        st.push(ret(sf.e()));
        st
    }));
    let xa = local("xs", anys.clone());
    let sa = local("s", any());
    d.push(define("zb_list_sum_any", &[&xa], any(), {
        let mut st = vec![
            sa.decl(call("zb_box_i64", vec![int(0)], any())),
            n.decl(len(xa.e())),
        ];
        st.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![sa.set(call(
                "zb_any_arith",
                vec![int(0), sa.e(), idx(xa.e(), i.e(), any())],
                any(),
            ))],
        ));
        st.push(ret(sa.e()));
        st
    }));

    let start = local("start", i64());
    let stop = local("stop", i64());
    let step = local("step", i64());
    let out = local("out", ints.clone());
    d.push(define(
        "zb_list_range",
        &[&start, &stop, &step],
        ints.clone(),
        vec![
            when(
                eq(step.e(), int(0)),
                vec![fatal("ValueError", text("range() arg 3 must not be zero"))],
            ),
            out.decl(list(Vec::new(), ints.clone())),
            i.decl(start.e()),
            if_(
                gt(step.e(), int(0)),
                vec![while_(
                    lt(i.e(), stop.e()),
                    vec![
                        expr(mcall(out.e(), "push", vec![i.e()], unit())),
                        i.set(add(i.e(), step.e())),
                    ],
                )],
                vec![while_(
                    gt(i.e(), stop.e()),
                    vec![
                        expr(mcall(out.e(), "push", vec![i.e()], unit())),
                        i.set(add(i.e(), step.e())),
                    ],
                )],
            ),
            ret(out.e()),
        ],
    ));

    // Every character of a string as its own string, walking the bytes
    // once.
    let text_in = local("s", string());
    let chars = local("out", strs.clone());
    let pos = local("pos", i64());
    let char_here = |s: Expr, pos: Expr| call("zb_str_char_at_byte", vec![s, pos], string());
    let next_pos = |s: Expr, pos: Expr| call("zb_str_next_byte", vec![s, pos], i64());
    d.push(define(
        "zb_str_chars",
        &[&text_in],
        strs.clone(),
        vec![
            chars.decl(list(Vec::new(), strs.clone())),
            n.decl(call("zb_str_len", vec![text_in.e()], i64())),
            pos.decl(int(0)),
            while_(
                lt(pos.e(), n.e()),
                vec![
                    expr(mcall(
                        chars.e(),
                        "push",
                        vec![char_here(text_in.e(), pos.e())],
                        unit(),
                    )),
                    pos.set(next_pos(text_in.e(), pos.e())),
                ],
            ),
            ret(chars.e()),
        ],
    ));
    // split on a separator: each piece is cut out by byte offset, and
    // the search resumes after the separator without copying the rest.
    let sep = local("sep", string());
    let at = local("at", i64());
    let w = local("w", i64());
    d.push(define(
        "zb_str_split",
        &[&text_in, &sep],
        strs.clone(),
        vec![
            chars.decl(list(Vec::new(), strs.clone())),
            w.decl(call("zb_str_len", vec![sep.e()], i64())),
            when(
                eq(w.e(), int(0)),
                vec![fatal("ValueError", text("empty separator"))],
            ),
            n.decl(call("zb_str_len", vec![text_in.e()], i64())),
            pos.decl(int(0)),
            at.decl(call(
                "zb_str_index_of_from",
                vec![text_in.e(), sep.e(), pos.e()],
                i64(),
            )),
            while_(
                ge(at.e(), int(0)),
                vec![
                    expr(mcall(
                        chars.e(),
                        "push",
                        vec![call(
                            "zb_str_bytes",
                            vec![text_in.e(), pos.e(), at.e()],
                            string(),
                        )],
                        unit(),
                    )),
                    pos.set(add(at.e(), w.e())),
                    at.set(call(
                        "zb_str_index_of_from",
                        vec![text_in.e(), sep.e(), pos.e()],
                        i64(),
                    )),
                ],
            ),
            expr(mcall(
                chars.e(),
                "push",
                vec![call(
                    "zb_str_bytes",
                    vec![text_in.e(), pos.e(), n.e()],
                    string(),
                )],
                unit(),
            )),
            ret(chars.e()),
        ],
    ));
    // split on runs of whitespace
    let c = local("c", string());
    let is_space = |c: Expr| call("zb_str_is_space", vec![c], boolean());
    d.push(define(
        "zb_str_is_space",
        &[&c],
        boolean(),
        vec![ret(or(
            or(
                call("zb_str_eq", vec![c.e(), text(" ")], boolean()),
                call("zb_str_eq", vec![c.e(), text("\t")], boolean()),
            ),
            or(
                call("zb_str_eq", vec![c.e(), text("\n")], boolean()),
                call("zb_str_eq", vec![c.e(), text("\r")], boolean()),
            ),
        ))],
    ));
    // Words are the runs between spaces, cut out of the text by byte
    // offset once each ends.
    let word_at = local("word_at", i64());
    let byte_slice = |s: Expr, a: Expr, b: Expr| call("zb_str_bytes", vec![s, a, b], string());
    d.push(define(
        "zb_str_split_ws",
        &[&text_in],
        strs.clone(),
        vec![
            chars.decl(list(Vec::new(), strs.clone())),
            n.decl(call("zb_str_len", vec![text_in.e()], i64())),
            pos.decl(int(0)),
            word_at.decl(int(-1)),
            while_(
                lt(pos.e(), n.e()),
                vec![
                    c.decl(char_here(text_in.e(), pos.e())),
                    if_(
                        is_space(c.e()),
                        vec![when(
                            ge(word_at.e(), int(0)),
                            vec![
                                expr(mcall(
                                    chars.e(),
                                    "push",
                                    vec![byte_slice(text_in.e(), word_at.e(), pos.e())],
                                    unit(),
                                )),
                                word_at.set(int(-1)),
                            ],
                        )],
                        vec![when(lt(word_at.e(), int(0)), vec![word_at.set(pos.e())])],
                    ),
                    pos.set(next_pos(text_in.e(), pos.e())),
                ],
            ),
            when(
                ge(word_at.e(), int(0)),
                vec![expr(mcall(
                    chars.e(),
                    "push",
                    vec![byte_slice(text_in.e(), word_at.e(), n.e())],
                    unit(),
                ))],
            ),
            ret(chars.e()),
        ],
    ));
    // One allocation for the whole result: the plugin reads the parts
    // straight out of the list's storage.
    let parts = borrowed("parts", strs.clone());
    d.push(extern_fn(
        "zb_str_join_raw",
        &[("data", i64()), ("n", i64()), ("sep", string())],
        string(),
        Some("$String$join_n"),
    ));
    d.push(define(
        "zb_str_join",
        &[&sep, &parts],
        string(),
        vec![ret(call(
            "zb_str_join_raw",
            vec![fld(parts.e(), "data", i64()), len(parts.e()), sep.e()],
            string(),
        ))],
    ));

    // Tuples: lists of dynamic values with their own tag and printing.
    let t = local("xs", anys.clone());
    d.push(extern_fn(
        "zb_box_tuple_raw",
        &[("xs", anys.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    d.push(extern_fn(
        "zb_unbox_tuple_raw",
        &[("x", any())],
        anys.clone(),
        Some("zyntax_box_pointer"),
    ));
    d.push(define(
        "zb_box_tuple",
        &[&t],
        any(),
        vec![ret(call(
            "zb_box_tuple_raw",
            vec![t.e(), int32(TUPLE_TAG as i32)],
            any(),
        ))],
    ));
    let x = local("x", any());
    let tag = local("tag", i64());
    d.push(define(
        "zb_unbox_tuple",
        &[&x],
        anys.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64())),
            when(
                ne(tag.e(), int(TUPLE_TAG)),
                vec![fatal(
                    "TypeError",
                    add(
                        text("expected a tuple, got "),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call("zb_unbox_tuple_raw", vec![x.e()], anys.clone())),
        ],
    ));
    d.push(define(
        "zb_tuple_repr",
        &[&t],
        string(),
        vec![
            when(
                eq(len(t.e()), int(1)),
                vec![ret(add(
                    add(
                        text("("),
                        call("zb_any_repr", vec![idx(t.e(), int(0), any())], string()),
                    ),
                    text(",)"),
                ))],
            ),
            ret(call(
                "zb_list_items_any",
                vec![t.e(), text("("), text(")")],
                string(),
            )),
        ],
    ));
    d
}
