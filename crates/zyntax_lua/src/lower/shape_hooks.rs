//! Shape hooks for Lua lowering.

use super::*;

/// The name of the program's function for a library hook.
pub(super) fn shape_hook_name(hook: &str) -> String {
    format!("lua$shape${hook}")
}

/// The hooks the library reads and writes slots through, over every
/// slotted shape of the program: each is a chain over the shapes'
/// numbers, then over the slots. In the order of
/// [`library::SHAPE_HOOKS`]; none when the program has no slotted
/// shape.
pub(super) fn shape_hooks(
    module: &Module<'_>,
    layouts: &[ShapeLayout],
    span: Span,
) -> Vec<TypedFunction> {
    if layouts.is_empty() {
        return Vec::new();
    }
    let table_t = module.ir(Ty::Table);
    let i64_t = prim(PrimitiveType::I64);
    let str_t = prim(PrimitiveType::String);
    let bool_t = prim(PrimitiveType::Bool);
    let t = || var(intern("t"), table_t.clone(), span);
    let i = || var(intern("i"), i64_t.clone(), span);
    let shape_is = |gid: i64| {
        binary(
            BinaryOp::Eq,
            field(t(), "shape", i64_t.clone(), span),
            int_lit(gid, span),
            bool_t.clone(),
            span,
        )
    };
    let slot_is = |bit: usize| {
        binary(
            BinaryOp::Eq,
            i(),
            int_lit(bit as i64, span),
            bool_t.clone(),
            span,
        )
    };
    let mut lowerer = Lowerer::new(module, CHUNK);
    let param = |name: &str, ty: Type| parameter(intern(name), ty, span);

    // zl_shape_index(t, name) -> i64
    let mut index_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            arms.push(if_(
                call(
                    "zb_str_eq",
                    vec![
                        var(intern("name"), str_t.clone(), span),
                        str_lit(&slot.name, span),
                    ],
                    bool_t.clone(),
                    span,
                ),
                vec![ret(Some(int_lit(slot.bit as i64, span)), span)],
                None,
                span,
            ));
        }
        arms.push(ret(Some(int_lit(-1, span)), span));
        index_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    index_body.push(ret(Some(int_lit(-1, span)), span));
    let index = typed_function(
        &shape_hook_name("index"),
        vec![param("t", table_t.clone()), param("name", str_t.clone())],
        i64_t.clone(),
        index_body,
        span,
    );

    // zl_shape_load(t, i) -> Any
    let mut load_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            let read = lowerer.slot_read(&t(), layout, slot, span);
            let boxed = lowerer.coerce(read, Ty::Any);
            let present = lowerer.slot_present(&t(), slot, span);
            arms.push(if_(
                slot_is(slot.bit),
                vec![ret(
                    Some(if_value(present, boxed, nil(span), Type::Any, span)),
                    span,
                )],
                None,
                span,
            ));
        }
        load_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    load_body.push(ret(Some(nil(span)), span));
    let load = typed_function(
        &shape_hook_name("load"),
        vec![param("t", table_t.clone()), param("i", i64_t.clone())],
        Type::Any,
        load_body,
        span,
    );

    // zl_shape_store(t, i, v) -> i64: 0 stored, 1 not of the slot's kind
    let v = || var(intern("v"), Type::Any, span);
    let category = || call("zb_any_category", vec![v()], i64_t.clone(), span);
    let is_cat = |c: i64| {
        binary(
            BinaryOp::Eq,
            category(),
            int_lit(c, span),
            bool_t.clone(),
            span,
        )
    };
    let mut store_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            // nil clears the slot's bit; nothing else is written.
            let mask_off = binary(
                BinaryOp::BitAnd,
                lowerer.present_of(&t(), span),
                int_lit(!slot.mask(), span),
                i64_t.clone(),
                span,
            );
            let mut on_nil = vec![assign(lowerer.present_of(&t(), span), mask_off, span)];
            if slot.kind == SlotKind::Any {
                let shaped = as_shape(t(), module.shape_ty(layout), span);
                on_nil.push(assign(
                    field(shaped, &format!("s{}", slot.bit), Type::Any, span),
                    nil(span),
                    span,
                ));
            }
            on_nil.push(ret(Some(int_lit(0, span)), span));
            let fits = match slot.kind {
                SlotKind::Int => binary(
                    BinaryOp::Or,
                    is_cat(library::INT),
                    is_cat(library::UINT),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Float => is_cat(library::FLOAT),
                SlotKind::Number => binary(
                    BinaryOp::Or,
                    binary(
                        BinaryOp::Or,
                        is_cat(library::INT),
                        is_cat(library::UINT),
                        bool_t.clone(),
                        span,
                    ),
                    is_cat(library::FLOAT),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Bool => is_cat(library::BOOL),
                SlotKind::Str => is_cat(library::STR),
                SlotKind::Table => binary(
                    BinaryOp::Eq,
                    cast(
                        call("zb_box_tag", vec![v()], prim(PrimitiveType::I32), span),
                        i64_t.clone(),
                        span,
                    ),
                    int_lit(library::table_tag(), span),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Scalar => binary(
                    BinaryOp::Lt,
                    category(),
                    int_lit(library::STR, span),
                    bool_t.clone(),
                    span,
                ),
                SlotKind::Any => bool_lit(true, span),
            };
            let stored = Val {
                node: v(),
                ty: Ty::Any,
            };
            let mut write = lowerer.slot_store(&t(), layout, slot, stored, false, span);
            write.push(ret(Some(int_lit(0, span)), span));
            arms.push(if_(
                slot_is(slot.bit),
                vec![
                    if_(
                        binary(BinaryOp::Eq, v(), nil(span), bool_t.clone(), span),
                        on_nil,
                        None,
                        span,
                    ),
                    if_(
                        fits,
                        write,
                        Some(vec![ret(Some(int_lit(1, span)), span)]),
                        span,
                    ),
                ],
                None,
                span,
            ));
        }
        store_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    store_body.push(ret(Some(int_lit(1, span)), span));
    let store = typed_function(
        &shape_hook_name("store"),
        vec![
            param("t", table_t.clone()),
            param("i", i64_t.clone()),
            param("v", Type::Any),
        ],
        i64_t.clone(),
        store_body,
        span,
    );

    // zl_shape_count(t) -> i64
    let mut count_body = Vec::new();
    for layout in layouts {
        count_body.push(if_(
            shape_is(layout.gid),
            vec![ret(Some(int_lit(layout.slots.len() as i64, span)), span)],
            None,
            span,
        ));
    }
    count_body.push(ret(Some(int_lit(0, span)), span));
    let count = typed_function(
        &shape_hook_name("count"),
        vec![param("t", table_t.clone())],
        i64_t.clone(),
        count_body,
        span,
    );

    // zl_shape_key(t, i) -> String
    let mut key_body = Vec::new();
    for layout in layouts {
        let mut arms = Vec::new();
        for slot in &layout.slots {
            arms.push(if_(
                slot_is(slot.bit),
                vec![ret(Some(str_lit(&slot.name, span)), span)],
                None,
                span,
            ));
        }
        key_body.push(if_(shape_is(layout.gid), arms, None, span));
    }
    key_body.push(ret(Some(str_lit("", span)), span));
    let key = typed_function(
        &shape_hook_name("key"),
        vec![param("t", table_t), param("i", i64_t.clone())],
        str_t,
        key_body,
        span,
    );
    vec![index, load, store, count, key]
}
