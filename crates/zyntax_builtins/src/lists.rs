//! Lists: the `List<T>` type, and the rules every language layers on the
//! type's own operations (`push`, `pop_last`, `insert_at`, `remove_at`,
//! `len`, indexing): negative indexes, errors, search, ordering,
//! printing, slicing. Built once per element kind.

use crate::build::*;
use crate::{list_of, Kind, Policy, TUPLE_TAG};
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

/// How one element kind compares and prints.
struct KindOps {
    kind: Kind,
    elem: Type,
    list: Type,
    /// `List<String>`, for building text piece by piece.
    strs: Type,
    eq: fn(Expr, Expr) -> Expr,
    lt: fn(Expr, Expr) -> Expr,
    repr: fn(Expr) -> Expr,
}

fn ops(kind: Kind, list_type: TypeId) -> KindOps {
    let elem = kind.ty();
    let (eq, lt, repr): (
        fn(Expr, Expr) -> Expr,
        fn(Expr, Expr) -> Expr,
        fn(Expr) -> Expr,
    ) = match kind {
        Kind::Int => (
            |a, b| eq(a, b),
            |a, b| lt(a, b),
            |x| call("zb_str_of_int", vec![x], string()),
        ),
        Kind::Float => (
            |a, b| eq(a, b),
            |a, b| lt(a, b),
            |x| call("zb_float_repr", vec![x], string()),
        ),
        Kind::Str => (
            |a, b| call("zb_str_eq", vec![a, b], boolean()),
            |a, b| call("zb_str_lt", vec![a, b], boolean()),
            |x| call("zb_str_repr", vec![x], string()),
        ),
        Kind::Any => (
            |a, b| call("zb_any_eq", vec![a, b], boolean()),
            |a, b| call("zb_any_lt", vec![a, b], boolean()),
            |x| call("zb_any_repr", vec![x], string()),
        ),
        // Instances compare by identity; ordering them is an error, and
        // printing one goes through its boxed form.
        Kind::Ptr => (
            |a, b| eq(a, b),
            |a, b| call("zb_ptr_lt", vec![a, b], boolean()),
            |x| {
                call(
                    "zb_any_repr",
                    vec![call("zb_hook_box_instance", vec![x], any())],
                    string(),
                )
            },
        ),
    };
    KindOps {
        kind,
        list: list_of(list_type, elem.clone()),
        strs: list_of(list_type, string()),
        elem,
        eq,
        lt,
        repr,
    }
}

fn len(xs: Expr) -> Expr {
    mcall(xs, "len", vec![], i64())
}

pub(crate) fn declarations(policy: &Policy, list_type: TypeId) -> Vec<Decl> {
    let mut out = vec![list_class()];
    for kind in Kind::ALL {
        out.extend(kind_declarations(&ops(kind, list_type)));
    }
    out.extend(shared(policy, list_type));
    out
}

fn kind_declarations(k: &KindOps) -> Vec<Decl> {
    let name = |op: &str| format!("zb_list_{op}_{}", k.kind.suffix());
    let xs = local("xs", k.list.clone());
    let ys = local("ys", k.list.clone());
    let out = local("out", k.list.clone());
    let i = local("i", i64());
    let j = local("j", i64());
    let n = local("n", i64());
    // The element a list keeps.
    let v = owned("v", k.elem.clone());
    let e = local("e", k.elem.clone());
    let what = local("what", string());
    let el = |xs: &Local, i: Expr| idx(xs.e(), i, k.elem.clone());
    let empty = || list(Vec::new(), k.list.clone());
    let mut d = Vec::new();

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
    d.push(define(
        &name("get"),
        &[&xs, &i],
        k.elem.clone(),
        vec![
            j.decl(norm(i.e(), "list index out of range")),
            ret(el(&xs, j.e())),
        ],
    ));
    d.push(define(
        &name("set"),
        &[&xs, &i, &v],
        unit(),
        vec![
            j.decl(norm(i.e(), "list assignment index out of range")),
            set_idx(xs.e(), j.e(), v.e()),
            ret_void(),
        ],
    ));
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
    // Insertion sort: stable, and short lists are the common case.
    d.push(define(
        &name("sort"),
        &[&xs],
        unit(),
        vec![
            n.decl(len(xs.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    v.decl(el(&xs, i.e())),
                    j.decl(sub(i.e(), int(1))),
                    while_(
                        and(ge(j.e(), int(0)), (k.lt)(v.e(), el(&xs, j.e()))),
                        vec![
                            set_idx(xs.e(), add(j.e(), int(1)), el(&xs, j.e())),
                            j.set(sub(j.e(), int(1))),
                        ],
                    ),
                    set_idx(xs.e(), add(j.e(), int(1)), v.e()),
                    i.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
    ));
    d.push(define(&name("extend"), &[&xs, &ys], unit(), {
        let mut s = vec![n.decl(len(ys.e()))];
        s.extend(for_range(
            &i,
            int(0),
            n.e(),
            vec![expr(mcall(xs.e(), "push", vec![el(&ys, i.e())], unit()))],
        ));
        s.push(ret_void());
        s
    }));
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
    let key_j = local("key_j", any());
    let moves = local("moves", boolean());
    d.push(define(
        &name("sort_by"),
        &[&xs, &keys, &descending],
        unit(),
        vec![
            n.decl(len(xs.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    v.decl(el(&xs, i.e())),
                    key.decl(idx(keys.e(), i.e(), any())),
                    j.decl(sub(i.e(), int(1))),
                    moves.decl(bool(true)),
                    while_(
                        and(ge(j.e(), int(0)), moves.e()),
                        vec![
                            key_j.decl(idx(keys.e(), j.e(), any())),
                            moves.set(if_expr(
                                descending.e(),
                                call("zb_any_lt", vec![key_j.e(), key.e()], boolean()),
                                call("zb_any_lt", vec![key.e(), key_j.e()], boolean()),
                            )),
                            when(
                                moves.e(),
                                vec![
                                    set_idx(xs.e(), add(j.e(), int(1)), el(&xs, j.e())),
                                    set_idx(
                                        keys.e(),
                                        add(j.e(), int(1)),
                                        idx(keys.e(), j.e(), any()),
                                    ),
                                    j.set(sub(j.e(), int(1))),
                                ],
                            ),
                        ],
                    ),
                    set_idx(xs.e(), add(j.e(), int(1)), v.e()),
                    set_idx(keys.e(), add(j.e(), int(1)), key.e()),
                    i.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
    ));
    // Descending order of the elements themselves, stable like `sort`.
    d.push(define(
        &name("sort_desc"),
        &[&xs],
        unit(),
        vec![
            n.decl(len(xs.e())),
            i.decl(int(1)),
            while_(
                lt(i.e(), n.e()),
                vec![
                    v.decl(el(&xs, i.e())),
                    j.decl(sub(i.e(), int(1))),
                    while_(
                        and(ge(j.e(), int(0)), (k.lt)(el(&xs, j.e()), v.e())),
                        vec![
                            set_idx(xs.e(), add(j.e(), int(1)), el(&xs, j.e())),
                            j.set(sub(j.e(), int(1))),
                        ],
                    ),
                    set_idx(xs.e(), add(j.e(), int(1)), v.e()),
                    i.add_assign(int(1)),
                ],
            ),
            ret_void(),
        ],
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
    // Everything boxed, for a list that becomes dynamic. A primitive
    // boxes as itself on the push; an instance address boxes as the
    // instance.
    let out_any = local("out", any_list.clone());
    let boxed = |e: Expr| match k.kind {
        Kind::Ptr => call("zb_hook_box_instance", vec![e], any()),
        _ => e,
    };
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
    let read = |e: Expr| match k.kind {
        Kind::Int => call("zb_any_as_i64", vec![e], i64()),
        Kind::Float => call("zb_any_as_f64", vec![e], f64()),
        Kind::Str => call("zb_any_as_str", vec![e], string()),
        Kind::Ptr => call("zb_hook_unbox_instance", vec![e, tag.e()], usize()),
        Kind::Any => e,
    };
    let from_params: Vec<&Local> = match k.kind {
        Kind::Ptr => vec![&anys_in, &tag],
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
        &format!("zb_box_list_raw_{}", k.kind.suffix()),
        &[("xs", k.list.clone()), ("tag", i32())],
        any(),
        Some("zyntax_box_ptr"),
    ));
    d.push(extern_fn(
        &format!("zb_unbox_list_raw_{}", k.kind.suffix()),
        &[("x", any())],
        k.list.clone(),
        Some("zyntax_box_pointer"),
    ));
    d.push(define(
        &name("box"),
        &[&xs],
        any(),
        vec![ret(call(
            &format!("zb_box_list_raw_{}", k.kind.suffix()),
            vec![xs.e(), int32(k.kind.list_tag() as i32)],
            any(),
        ))],
    ));
    let x = local("x", any());
    let tag = local("tag", i64());
    d.push(define(
        &name("unbox"),
        &[&x],
        k.list.clone(),
        vec![
            tag.decl(cast(call("zb_box_tag", vec![x.e()], i32()), i64())),
            when(
                ne(tag.e(), int(k.kind.list_tag())),
                vec![fatal(
                    "TypeError",
                    add(
                        text("expected a list, got "),
                        call("zb_any_type", vec![x.e()], string()),
                    ),
                )],
            ),
            ret(call(
                &format!("zb_unbox_list_raw_{}", k.kind.suffix()),
                vec![x.e()],
                k.list.clone(),
            )),
        ],
    ));
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
    // split on a separator
    let sep = local("sep", string());
    let rest = local("rest", string());
    let at = local("at", i64());
    let w = local("w", i64());
    d.push(define(
        "zb_str_split",
        &[&text_in, &sep],
        strs.clone(),
        vec![
            chars.decl(list(Vec::new(), strs.clone())),
            when(
                eq(call("zb_str_chars_len", vec![sep.e()], i64()), int(0)),
                vec![fatal("ValueError", text("empty separator"))],
            ),
            rest.decl(text_in.e()),
            at.decl(call("zb_str_index_of", vec![rest.e(), sep.e()], i64())),
            w.decl(call("zb_str_chars_len", vec![sep.e()], i64())),
            while_(
                ge(at.e(), int(0)),
                vec![
                    expr(mcall(
                        chars.e(),
                        "push",
                        vec![call(
                            "zb_str_substring",
                            vec![rest.e(), int(0), at.e()],
                            string(),
                        )],
                        unit(),
                    )),
                    rest.set(call(
                        "zb_str_substring",
                        vec![
                            rest.e(),
                            add(at.e(), w.e()),
                            call("zb_str_chars_len", vec![rest.e()], i64()),
                        ],
                        string(),
                    )),
                    at.set(call("zb_str_index_of", vec![rest.e(), sep.e()], i64())),
                ],
            ),
            expr(mcall(chars.e(), "push", vec![rest.e()], unit())),
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
    let parts = local("parts", strs.clone());
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
