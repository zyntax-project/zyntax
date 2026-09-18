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
        // The same box is equal to itself before its value is looked
        // at, which is how a membership test or an element comparison
        // treats identity; shared boxes of small integers make that the
        // common case.
        Kind::Any => (
            |a, b| {
                or(
                    eq(a.clone(), b.clone()),
                    call("zb_any_eq", vec![a, b], boolean()),
                )
            },
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
    for kind in Kind::ALL {
        out.extend(kind_declarations(&ops(kind, list_type)));
    }
    out.extend(shared(policy, list_type));
    out
}

fn kind_declarations(k: &KindOps) -> Vec<Decl> {
    let name = |op: &str| format!("zb_list_{op}_{}", k.kind.suffix());
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
    let checked_index = |msg: &str| {
        if k.kind != Kind::Float {
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
        &[&carried],
        any(),
        vec![ret(call(
            &format!("zb_box_list_raw_{}", k.kind.suffix()),
            vec![carried.e(), int32(k.kind.list_tag() as i32)],
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
        Kind::Any => {
            for other in Kind::ALL.iter().filter(|o| **o != Kind::Any) {
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
        }
        Kind::Int | Kind::Float | Kind::Str => {
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
        Kind::Ptr => {}
    }
    unbox.push(when(
        ne(tag.e(), int(k.kind.list_tag())),
        vec![fatal(
            "TypeError",
            add(
                text("expected a list, got "),
                call("zb_any_type", vec![x.e()], string()),
            ),
        )],
    ));
    unbox.push(ret(call(
        &format!("zb_unbox_list_raw_{}", k.kind.suffix()),
        vec![x.e()],
        k.list.clone(),
    )));
    d.push(define(&name("unbox"), &[&x], k.list.clone(), unbox));
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
