//! Unary for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn unary_op(&mut self, op: &UnOp, v: Val, desc: Desc, span: Span) -> Result<Val> {
        let descs = Described::operand(desc);
        Ok(match op {
            UnOp::Not(_) => {
                let t = self.truthy(v);
                Val {
                    node: unary(UnaryOp::Not, t, prim(PrimitiveType::Bool), span),
                    ty: Ty::Bool,
                }
            }
            UnOp::Minus(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::Sub,
                        int_lit(0, span),
                        v.node,
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                Ty::Float => Val {
                    node: unary(UnaryOp::Minus, v.node, prim(PrimitiveType::F64), span),
                    ty: Ty::Float,
                },
                Ty::Number | Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => {
                    let mut pre = Vec::new();
                    let held = self.hold(v.clone(), &mut pre);
                    let n = NumberParts::of(&held.node, span);
                    // A scalar that is not a number raises, as the
                    // dynamic value does.
                    if v.ty != Ty::Number {
                        let boxed = self.coerce(held.clone(), Ty::Any);
                        pre.push(if_(
                            n.is_numeric(span),
                            Vec::new(),
                            Some(vec![
                                expr_stmt(call("zl_unm", vec![boxed], Type::Any, span)),
                                self.pending_check_described(span, &descs),
                            ]),
                            span,
                        ));
                    }
                    let is_int = n.is_int(span);
                    let int = binary(
                        BinaryOp::Sub,
                        int_lit(0, span),
                        n.int,
                        prim(PrimitiveType::I64),
                        span,
                    );
                    let int = self
                        .hold(
                            Val {
                                node: int,
                                ty: Ty::Int,
                            },
                            &mut pre,
                        )
                        .node;
                    let float = float_word(
                        is_int,
                        int.clone(),
                        unary(UnaryOp::Minus, n.float, prim(PrimitiveType::F64), span),
                        span,
                    );
                    Val {
                        node: block_value(pre, number_value(n.tag, int, float, span), span),
                        ty: Ty::Number,
                    }
                }
                Ty::Shape(_)
                    if let Some(sides) = self.typer().metamethod_sides("__unm", v.ty, v.ty)
                        && !sides.is_empty() =>
                {
                    // The one operand is both: the handler takes it twice.
                    let sides: Vec<(bool, ShapeId, FuncId)> =
                        sides.into_iter().filter(|(left, _, _)| *left).collect();
                    let mut pre = Vec::new();
                    let held = self.hold(v, &mut pre);
                    let b = self.boxed(held.clone());
                    let generic = self.guard_described(
                        Val {
                            node: call("zl_unm", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    let ty = self
                        .typer()
                        .metamethod_result("__unm", held.ty, held.ty)
                        .unwrap_or(Ty::Any)
                        .settled();
                    let value = self.metamethod_dispatch(
                        "__unm",
                        sides,
                        vec![held.clone(), held],
                        generic,
                        ty,
                        span,
                    );
                    Val {
                        node: block_value(pre, value, span),
                        ty,
                    }
                }
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_unm", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            UnOp::Hash(_) => match v.ty {
                Ty::Str => Val {
                    node: call("zb_str_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                Ty::Table if !self.scopes().len_meta => Val {
                    node: call("zl_table_len", vec![v.node], prim(PrimitiveType::I64), span),
                    ty: Ty::Int,
                },
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does. With no `__len` in
                // any metatable it may have, its length is its array
                // part's.
                Ty::Shape(k) if !self.scopes().len_meta => {
                    let i64_t = prim(PrimitiveType::I64);
                    let mut pre = Vec::new();
                    let t = self.hold(v, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let on_nil = self.guard_described(
                        Val {
                            node: call("zl_len_any", vec![nil(span)], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    let on_nil = self.coerce(on_nil, Ty::Int);
                    let len = if self.m.inferred.plain_len(k) {
                        list_len(
                            call(
                                "zb_unbox_list_raw_any",
                                vec![field(t.node, "arr", Type::Any, span)],
                                self.m.anys(),
                                span,
                            ),
                            span,
                        )
                    } else {
                        call("zl_table_len", vec![t.node], i64_t.clone(), span)
                    };
                    Val {
                        node: block_value(pre, if_value(is_null, on_nil, len, i64_t, span), span),
                        ty: Ty::Int,
                    }
                }
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_len_any", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            UnOp::Tilde(_) => match v.ty {
                Ty::Int => Val {
                    node: binary(
                        BinaryOp::BitXor,
                        v.node,
                        int_lit(-1, span),
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    ty: Ty::Int,
                },
                _ => {
                    let b = self.boxed(v);
                    self.guard_described(
                        Val {
                            node: call("zl_bnot", vec![b], Type::Any, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    )
                }
            },
            _ => return unsupported("this unary operator", span),
        })
    }

    /// An operator's handler, called directly: for each side that is a
    /// shaped table and each of its classes, in the order the runtime
    /// tries them, when the operand's metatable is that class and the
    /// handler is present in it; else `generic`, the runtime's own
    /// dispatch. The operands are held; every path yields `ty`.
    pub(super) fn metamethod_dispatch(
        &mut self,
        event: &str,
        sides: Vec<(bool, ShapeId, FuncId)>,
        operands: Vec<Val>,
        generic: Val,
        ty: Ty,
        span: Span,
    ) -> Node {
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let metamethod = Some(format!("metamethod '{}'", event.trim_start_matches("__")));
        let mut pre = Vec::new();
        let mut value = self.coerce(generic, ty);
        for (left, class, f) in sides.into_iter().rev() {
            let Some((layout, slot)) = self.slot_of(class, event) else {
                continue;
            };
            let operand = if left { &operands[0] } else { &operands[1] };
            let not_null = |x: &Node| {
                binary(
                    BinaryOp::Ne,
                    x.clone(),
                    null(table_t.clone(), span),
                    bool_t.clone(),
                    span,
                )
            };
            let meta = self.hold(
                Val {
                    node: if_value(
                        not_null(&operand.node),
                        field(operand.node.clone(), "meta", table_t.clone(), span),
                        null(table_t.clone(), span),
                        table_t.clone(),
                        span,
                    ),
                    ty: Ty::Table,
                },
                &mut pre,
            );
            // A handler stored where every path here has run needs no
            // presence test; a table whose one metatable is this class
            // from birth on needs only to be a table.
            let known = self.known_present(class, event);
            let only_class = match operand.ty {
                Ty::Shape(k) => {
                    let inferred = self.m.inferred;
                    inferred.meta_never_nil(k)
                        && inferred.shape(k).classes.iter().eq(std::iter::once(&class))
                }
                _ => false,
            };
            let ok = if known && only_class {
                not_null(&operand.node)
            } else {
                let shape_is = binary(
                    BinaryOp::Eq,
                    field(meta.node.clone(), "shape", i64_t.clone(), span),
                    int_lit(layout.gid, span),
                    bool_t.clone(),
                    span,
                );
                let is_class = if known {
                    shape_is
                } else {
                    binary(
                        BinaryOp::And,
                        shape_is,
                        self.slot_present(&meta.node, &slot, span),
                        bool_t.clone(),
                        span,
                    )
                };
                if_value(
                    not_null(&meta.node),
                    is_class,
                    bool_lit(false, span),
                    bool_t.clone(),
                    span,
                )
            };
            let ok = self.hold(
                Val {
                    node: ok,
                    ty: Ty::Bool,
                },
                &mut pre,
            );
            let record = self.slot_read(&meta.node, layout, &slot, span);
            let multi = self.direct_call_vals(
                f,
                Some(record),
                Vec::new(),
                operands.clone(),
                None,
                &metamethod,
                span,
            );
            let first = match multi {
                Multi::Fixed(vals) => vals
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| self.nil_val(span)),
                Multi::Dynamic(node) => Val {
                    node: call("zl_first", vec![node], Type::Any, span),
                    ty: Ty::Any,
                },
                Multi::None(node) => Val {
                    node: block_value(vec![expr_stmt(node)], nil(span), span),
                    ty: Ty::Nil,
                },
            };
            // Held: the call's check leaves its block, and a branch's
            // value is read after it.
            let mut arm_pre = Vec::new();
            let arm = self.coerce(first, ty);
            let arm = self.hold(Val { node: arm, ty }, &mut arm_pre);
            value = if_value(
                ok.node,
                block_value(arm_pre, arm.node, span),
                value,
                self.ir(ty),
                span,
            );
        }
        block_value(pre, value, span)
    }
}
