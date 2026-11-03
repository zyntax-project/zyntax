//! Field write for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// Whether a value of its type is nil.
    pub(super) fn is_nil_val(&mut self, v: Val) -> Node {
        let span = v.node.span;
        let bool_t = prim(PrimitiveType::Bool);
        match v.ty {
            Ty::Nil => block_value(vec![expr_stmt(v.node)], bool_lit(true, span), span),
            Ty::Bool | Ty::Int | Ty::Float | Ty::Number | Ty::Str | Ty::Table => {
                if Self::is_simple(&v.node) {
                    bool_lit(false, span)
                } else {
                    block_value(vec![expr_stmt(v.node)], bool_lit(false, span), span)
                }
            }
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.has_tag(TAG_NIL, span), span)
            }
            Ty::Shape(_) => binary(
                BinaryOp::Eq,
                v.node,
                null(self.ir(Ty::Table), span),
                bool_t,
                span,
            ),
            Ty::Func(_) | Ty::Any | Ty::Unknown => {
                binary(BinaryOp::Eq, v.node, nil(span), bool_t, span)
            }
        }
    }

    /// `obj[key] = value`, with `__newindex`; `desc` is what `obj` is
    /// called.
    pub(super) fn index_write(
        &mut self,
        obj: Val,
        key: Val,
        value: Val,
        desc: Desc,
        span: Span,
    ) -> St {
        let descs = Described::operand(desc);
        let unit = prim(PrimitiveType::Unit);
        // A shaped receiver's slot takes the value when the field is
        // there already or no metatable could take it instead; else
        // the dynamic path, which raises for nil and asks `__newindex`.
        if let (Ty::Shape(shape), Some(name)) = (obj.ty, self.literal_key(&key)) {
            let found = self.m.layout(shape).and_then(|layout| {
                layout
                    .slots
                    .iter()
                    .find(|s| s.name == name)
                    .map(|s| (layout, s))
            });
            if let Some((layout, slot)) = found {
                let present = self
                    .value_fields(&obj)
                    .is_some_and(|fields| fields.contains(&name));
                let bool_t = prim(PrimitiveType::Bool);
                let mut pre = Vec::new();
                let t = self.hold(obj, &mut pre);
                let stored = self.slot_value(value, slot);
                let stored = self.hold(stored, &mut pre);
                let not_null = binary(
                    BinaryOp::Ne,
                    t.node.clone(),
                    null(self.ir(Ty::Table), span),
                    bool_t.clone(),
                    span,
                );
                // No metatable the types know of for this shape holds a
                // `__newindex`: nothing can take the store instead.
                let info = self.m.inferred.shape(shape);
                let no_newindex = !info.unknown_meta
                    && info
                        .classes
                        .iter()
                        .all(|c| self.m.inferred.shape(*c).field("__newindex").is_none());
                let fast = if no_newindex {
                    not_null
                } else {
                    let no_meta = binary(
                        BinaryOp::Eq,
                        field(t.node.clone(), "meta", self.ir(Ty::Table), span),
                        null(self.ir(Ty::Table), span),
                        bool_t.clone(),
                        span,
                    );
                    binary(
                        BinaryOp::And,
                        not_null,
                        binary(
                            BinaryOp::Or,
                            self.slot_present(&t.node, slot, span),
                            no_meta,
                            bool_t.clone(),
                            span,
                        ),
                        bool_t,
                        span,
                    )
                };
                // A field every table of the shape is born with, that
                // nothing stores nil in and no store under a computed
                // key reaches, keeps its bit set.
                let settled = present || (info.always_present(&name) && !info.dynamic_keys);
                let then = self.slot_store(&t.node, layout, slot, stored.clone(), settled, span);
                if present {
                    pre.extend(then);
                    return stmt(
                        TypedStatement::Block(TypedBlock {
                            statements: pre,
                            span,
                        }),
                        span,
                    );
                }
                let els = if no_newindex {
                    // Only a nil receiver gets here: it raises, on the
                    // cold path a read takes.
                    let mut raise = vec![
                        self.set_line(span),
                        expr_stmt(call("zl_index_nil", vec![], Type::Any, span)),
                    ];
                    raise.extend(self.leave_described(span, &descs));
                    raise
                } else {
                    let boxed = self.coerce(stored, Ty::Any);
                    let boxed_t = self.coerce(t, Ty::Any);
                    let k = self.constant_key(&key).expect("a constant key");
                    vec![
                        expr_stmt(call("zl_setindex_key", vec![boxed_t, k, boxed], unit, span)),
                        self.pending_check_described(span, &descs),
                    ]
                };
                pre.push(if_(fast, then, Some(els), span));
                return stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: pre,
                        span,
                    }),
                    span,
                );
            }
        }
        let v = self.boxed(value);
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_setindex_key", vec![obj.node, k, v], unit, span),
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does.
                Ty::Shape(_) => {
                    let mut pre = Vec::new();
                    let t = self.hold(obj, &mut pre);
                    let held_v = self.hold(
                        Val {
                            node: v,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    );
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let store = if_(
                        is_null,
                        vec![expr_stmt(call(
                            "zl_setindex_key",
                            vec![nil(span), k.clone(), held_v.node.clone()],
                            unit.clone(),
                            span,
                        ))],
                        Some(vec![expr_stmt(call(
                            "zl_table_setindex_key",
                            vec![t.node, k, held_v.node],
                            unit,
                            span,
                        ))]),
                        span,
                    );
                    pre.push(store);
                    pre.push(self.pending_check_described(span, &descs));
                    return stmt(
                        TypedStatement::Block(TypedBlock {
                            statements: pre,
                            span,
                        }),
                        span,
                    );
                }
                _ => {
                    let o = self.boxed(obj);
                    call("zl_setindex_key", vec![o, k, v], unit, span)
                }
            };
            return self.guarded_stmt(node, &descs);
        }
        // A shaped receiver is a table unless it is nil, when the dynamic
        // path raises as Lua does.
        if let Ty::Shape(_) = obj.ty {
            let table_t = self.ir(Ty::Table);
            let mut pre = Vec::new();
            let t = self.hold(obj, &mut pre);
            let held_v = self
                .hold(
                    Val {
                        node: v,
                        ty: Ty::Any,
                    },
                    &mut pre,
                )
                .node;
            let (on_table, on_nil) = match key.ty {
                Ty::Int => {
                    let k = self.hold(key, &mut pre);
                    (
                        call(
                            "zl_table_seti",
                            vec![t.node.clone(), k.node.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_seti",
                            vec![nil(span), k.node, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
                Ty::Str => {
                    let k = self.hold(key, &mut pre);
                    (
                        call(
                            "zl_table_setindex_str",
                            vec![t.node.clone(), k.node.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_setindex_str",
                            vec![nil(span), k.node, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
                _ => {
                    let k = self.boxed(key);
                    let k = self
                        .hold(
                            Val {
                                node: k,
                                ty: Ty::Any,
                            },
                            &mut pre,
                        )
                        .node;
                    (
                        call(
                            "zl_table_setindex",
                            vec![t.node.clone(), k.clone(), held_v.clone()],
                            unit.clone(),
                            span,
                        ),
                        call(
                            "zl_setindex",
                            vec![nil(span), k, held_v],
                            unit.clone(),
                            span,
                        ),
                    )
                }
            };
            let is_null = binary(
                BinaryOp::Eq,
                t.node,
                null(table_t, span),
                prim(PrimitiveType::Bool),
                span,
            );
            pre.push(if_(
                is_null,
                vec![expr_stmt(on_nil)],
                Some(vec![expr_stmt(on_table)]),
                span,
            ));
            pre.push(self.pending_check_described(span, &descs));
            return stmt(
                TypedStatement::Block(TypedBlock {
                    statements: pre,
                    span,
                }),
                span,
            );
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_seti", vec![obj.node, key.node, v], unit, span),
                Ty::Str => call(
                    "zl_table_setindex_str",
                    vec![obj.node, key.node, v],
                    unit,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_setindex", vec![obj.node, k, v], unit, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_seti", vec![o, key.node, v], unit, span),
                    Ty::Str => call("zl_setindex_str", vec![o, key.node, v], unit, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_setindex", vec![o, k, v], unit, span)
                    }
                }
            }
        };
        self.guarded_stmt(node, &descs)
    }
}
