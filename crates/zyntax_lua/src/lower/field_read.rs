//! Field read for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The name a constant string key spells.
    pub(super) fn literal_key(&self, key: &Val) -> Option<String> {
        match &key.node.node {
            TypedExpression::Literal(TypedLiteral::String(s)) if key.ty == Ty::Str => {
                s.resolve_global()
            }
            _ => None,
        }
    }

    /// A name the program spells, boxed once and shared: the compiler
    /// keeps a boxed string literal as one constant.
    pub(super) fn constant_key(&mut self, key: &Val) -> Option<Node> {
        match &key.node.node {
            TypedExpression::Literal(TypedLiteral::String(_)) if key.ty == Ty::Str => {
                Some(self.boxed(key.clone()))
            }
            _ => None,
        }
    }

    /// `obj.name`: a read whose result the types may know, through a
    /// shaped receiver; from the slot when the receiver's shape has one
    /// for the name and it holds a value, else the dynamic path, which
    /// raises for nil and consults the metatable for an absent field.
    pub(super) fn field_read(
        &mut self,
        obj: Val,
        key: Val,
        name: &str,
        desc: Desc,
        span: Span,
    ) -> Val {
        let Ty::Shape(k) = obj.ty else {
            return self.index_read(obj, key, desc, span);
        };
        if self
            .value_fields(&obj)
            .is_some_and(|fields| fields.contains(name))
            && let Some((layout, slot)) = self.slot_of(k, name)
            && let Some((_, ty)) = self.m.inferred.shape(k).field(name)
        {
            let mut pre = Vec::new();
            let obj = self.hold(obj, &mut pre);
            let ty = ty.settled();
            let value = self.slot_read_as(&obj.node, layout, &slot, ty, span);
            let value = self.coerce(value, ty);
            return Val {
                node: block_value(pre, value, span),
                ty,
            };
        }
        let ty = self.typer().field_ty(Ty::Shape(k), name).settled();
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let t = self.hold(obj, &mut pre);
        let not_null = |x: &Node| {
            binary(
                BinaryOp::Ne,
                x.clone(),
                null(table_t.clone(), span),
                bool_t.clone(),
                span,
            )
        };
        let info = self.m.inferred.shape(k);
        let own = info.field(name).map(|(_, ty)| ty);
        let sure = info.always_present(name);
        let slot = if own.is_some() {
            self.slot_of(k, name)
        } else {
            None
        };
        let finder = if sure {
            None
        } else {
            self.finder(k, name, FinderMode::Table, span)
        };
        // Whether every place the field may be found is checked here:
        // the table's own slot, and the finder for each lookup that
        // goes through a metatable. Then what nothing finds is nil.
        let closed = !sure
            && slot.is_some()
            && self.m.inferred.lookups(k, name).is_some_and(|lookups| {
                finder.is_some() || lookups.iter().all(|l| l.hops.is_empty())
            });
        // Where nothing below finds the field. A slot the table always
        // holds is only missed by a nil receiver, which raises; so is
        // a closed lookup, which else gives nil; anything else takes
        // the general read, which raises for nil and consults a
        // metatable the types do not follow.
        let slot_desc = desc.clone();
        let mut value = if sure && slot.is_some() {
            self.index_nil(desc, ty, span)
        } else if closed {
            let raise = self.index_nil(desc, ty, span);
            let absent = self.coerce(
                Val {
                    node: nil(span),
                    ty: Ty::Nil,
                },
                ty,
            );
            if_value(not_null(&t.node), absent, raise, self.ir(ty), span)
        } else {
            let general = self.index_read(t.clone(), key, desc, span);
            self.coerce(general, ty)
        };
        // Then through the metatables, by the finder.
        if let Some(finder) = finder {
            let found = self.hold(
                Val {
                    node: call(&finder.helper, vec![t.node.clone()], table_t.clone(), span),
                    ty: Ty::Table,
                },
                &mut pre,
            );
            let read = self.found_value(&finder, &found.node, name, ty, span);
            value = if_value(not_null(&found.node), read, value, self.ir(ty), span);
        }
        // The table's own slot first.
        if let Some((layout, slot)) = &slot
            && sure
            && slot.kind == SlotKind::Number
            && ty == Ty::Number
        {
            value = self.sure_number_read(&t.node, layout, slot, slot_desc, span);
        } else if let Some((layout, slot)) = slot {
            let fast = if sure {
                not_null(&t.node)
            } else {
                binary(
                    BinaryOp::And,
                    not_null(&t.node),
                    self.slot_present(&t.node, &slot, span),
                    bool_t,
                    span,
                )
            };
            let from_slot = self.slot_read_as(&t.node, layout, &slot, ty, span);
            let from_slot = self.coerce(from_slot, ty);
            value = if_value(fast, from_slot, value, self.ir(ty), span);
        }
        Val {
            node: block_value(pre, value, span),
            ty,
        }
    }

    /// Number slot `slot` of table `t`, which every table of its shape
    /// holds: a nil `t` raises in computing the tag, ahead of the two
    /// words' reads, so the number is built from its parts with no
    /// merge of whole numbers.
    pub(super) fn sure_number_read(
        &mut self,
        t: &Node,
        layout: &ShapeLayout,
        slot: &Slot,
        desc: Desc,
        span: Span,
    ) -> Node {
        let read = self.number_slot_read(t, layout, slot, Ty::Number, span);
        let TypedExpression::Tuple(parts) = read.node.node else {
            unreachable!("a number slot reads as its three parts")
        };
        let [tag, int, float]: [Node; 3] = parts.try_into().expect("three parts");
        let not_null = binary(
            BinaryOp::Ne,
            t.clone(),
            null(self.ir(Ty::Table), span),
            prim(PrimitiveType::Bool),
            span,
        );
        let raise = self.index_nil(desc, Ty::Int, span);
        let tag = if_value(not_null, tag, raise, prim(PrimitiveType::I64), span);
        number_value(tag, int, float, span)
    }

    /// Indexing a nil receiver: the error, raised and left through,
    /// then a value of `ty` no one reads. The raise always leaves an
    /// error pending, so nothing past it runs.
    pub(super) fn index_nil(&mut self, desc: Desc, ty: Ty, span: Span) -> Node {
        let descs = Described::operand(desc);
        let leave = self.leave_described(span, &descs);
        let raise = vec![
            expr_stmt(call("zl_index_nil", vec![], Type::Any, span)),
            if_(bool_lit(true, span), leave, None, span),
        ];
        let unread = match ty {
            Ty::Table => null(self.ir(Ty::Table), span),
            _ => self.zero_of(ty, span),
        };
        block_value(raise, unread, span)
    }

    /// `obj[key]`, with `__index`; `desc` is what `obj` is called.
    pub(super) fn index_read(&mut self, obj: Val, key: Val, desc: Desc, span: Span) -> Val {
        let descs = Described::operand(desc);
        if let Some(k) = self.constant_key(&key) {
            let node = match obj.ty {
                Ty::Table => call("zl_table_index_key", vec![obj.node, k], Type::Any, span),
                // A shaped table is a table unless it is nil, when the
                // dynamic path raises as Lua does.
                Ty::Shape(_) => {
                    let mut pre = Vec::new();
                    let t = self.hold(obj, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(self.ir(Ty::Table), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    let read = if_value(
                        is_null,
                        call("zl_index_key", vec![nil(span), k.clone()], Type::Any, span),
                        call("zl_table_index_key", vec![t.node, k], Type::Any, span),
                        Type::Any,
                        span,
                    );
                    let v = self.guard_described(
                        Val {
                            node: block_value(pre, read, span),
                            ty: Ty::Any,
                        },
                        &descs,
                    );
                    return v;
                }
                _ => {
                    let o = self.boxed(obj);
                    call("zl_index_key", vec![o, k], Type::Any, span)
                }
            };
            return self.guarded_described(Val { node, ty: Ty::Any }, &descs);
        }
        if let Ty::Shape(s) = obj.ty {
            return self.shape_index_read(obj, s, key, &descs, span);
        }
        let node = match obj.ty {
            Ty::Table => match key.ty {
                Ty::Int => call("zl_table_geti", vec![obj.node, key.node], Type::Any, span),
                Ty::Str => call(
                    "zl_table_index_str",
                    vec![obj.node, key.node],
                    Type::Any,
                    span,
                ),
                _ => {
                    let k = self.boxed(key);
                    call("zl_table_index", vec![obj.node, k], Type::Any, span)
                }
            },
            _ => {
                let o = self.boxed(obj);
                match key.ty {
                    Ty::Int => call("zl_geti", vec![o, key.node], Type::Any, span),
                    Ty::Str => call("zl_index_str", vec![o, key.node], Type::Any, span),
                    _ => {
                        let k = self.boxed(key);
                        call("zl_index", vec![o, k], Type::Any, span)
                    }
                }
            }
        };
        self.guarded_described(Val { node, ty: Ty::Any }, &descs)
    }

    /// `obj[key]` for `obj` a table of shape `s` and a key that is not a
    /// constant string. Under an integer, when the types say what the
    /// array part holds and no metatable could answer instead, an
    /// element in range is read in place; everything else takes the
    /// general path, which raises for a nil receiver.
    pub(super) fn shape_index_read(
        &mut self,
        obj: Val,
        s: ShapeId,
        key: Val,
        descs: &Described,
        span: Span,
    ) -> Val {
        let table_t = self.ir(Ty::Table);
        let bool_t = prim(PrimitiveType::Bool);
        let i64_t = prim(PrimitiveType::I64);
        let mut pre = Vec::new();
        let t = self.hold(obj, &mut pre);
        let k = self.hold(key, &mut pre);
        let is_null = binary(
            BinaryOp::Eq,
            t.node.clone(),
            null(table_t.clone(), span),
            bool_t.clone(),
            span,
        );
        let (on_table, on_nil) = match k.ty {
            Ty::Int => (
                call(
                    "zl_table_geti",
                    vec![t.node.clone(), k.node.clone()],
                    Type::Any,
                    span,
                ),
                call("zl_geti", vec![nil(span), k.node.clone()], Type::Any, span),
            ),
            Ty::Str => (
                call(
                    "zl_table_index_str",
                    vec![t.node.clone(), k.node.clone()],
                    Type::Any,
                    span,
                ),
                call(
                    "zl_index_str",
                    vec![nil(span), k.node.clone()],
                    Type::Any,
                    span,
                ),
            ),
            _ => {
                let boxed = self.boxed(k.clone());
                let boxed = self
                    .hold(
                        Val {
                            node: boxed,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    )
                    .node;
                (
                    call(
                        "zl_table_index",
                        vec![t.node.clone(), boxed.clone()],
                        Type::Any,
                        span,
                    ),
                    call("zl_index", vec![nil(span), boxed], Type::Any, span),
                )
            }
        };
        let general = self.guard_described(
            Val {
                node: if_value(is_null, on_nil, on_table, Type::Any, span),
                ty: Ty::Any,
            },
            descs,
        );
        let ty = self.m.inferred.element_read_ty(s, k.ty).settled();
        // Without a metatable to ask, an element in range is the
        // answer whatever its kind; a float key names an element only
        // when it is integral, and the general path decides that.
        let info = self.m.inferred.shape(s);
        let plain = !info.unknown_meta && info.classes.is_empty();
        if !plain {
            let node = if ty == Ty::Any {
                general.node
            } else {
                self.coerce(general, ty)
            };
            return Val {
                node: block_value(pre, node, span),
                ty,
            };
        }
        if k.ty != Ty::Int {
            let general = self.coerce(general, ty);
            return Val {
                node: block_value(pre, general, span),
                ty,
            };
        }
        // The array part: `1..len` holds elements `0..len - 1`.
        let arr = || {
            call(
                "zb_unbox_list_raw_any",
                vec![field(t.node.clone(), "arr", Type::Any, span)],
                self.m.anys(),
                span,
            )
        };
        let in_range = binary(
            BinaryOp::And,
            binary(
                BinaryOp::And,
                binary(
                    BinaryOp::Ne,
                    t.node.clone(),
                    null(table_t, span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Ge,
                    k.node.clone(),
                    int_lit(1, span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Le,
                k.node.clone(),
                list_len(arr(), span),
                bool_t.clone(),
                span,
            ),
            bool_t.clone(),
            span,
        );
        let slot = binary(
            BinaryOp::Sub,
            k.node.clone(),
            int_lit(1, span),
            i64_t.clone(),
            span,
        );
        let element = index(arr(), slot.clone(), Type::Any, span);
        let element = self.coerce(
            Val {
                node: element,
                ty: Ty::Any,
            },
            ty,
        );
        // A flat array of integers is read in place: raw when the type
        // says integer, boxed otherwise.
        let flat = index(
            call(
                "zb_unbox_list_raw_i64",
                vec![field(t.node.clone(), "arr", Type::Any, span)],
                self.m.types.ints(),
                span,
            ),
            slot,
            i64_t.clone(),
            span,
        );
        let flat = if ty == Ty::Int {
            flat
        } else {
            let boxed = call("zb_box_i64", vec![flat], Type::Any, span);
            self.coerce(
                Val {
                    node: boxed,
                    ty: Ty::Any,
                },
                ty,
            )
        };
        let is_flat = binary(
            BinaryOp::Eq,
            cast(
                call(
                    "zb_box_header_tag",
                    vec![field(t.node.clone(), "arr", Type::Any, span)],
                    prim(PrimitiveType::I32),
                    span,
                ),
                i64_t,
                span,
            ),
            int_lit(crate::library::int_list_tag(), span),
            bool_t.clone(),
            span,
        );
        // An array the types say holds only values of another kind is
        // never flat.
        let may_be_flat = !matches!(
            ty,
            Ty::Str | Ty::Float | Ty::Bool | Ty::Table | Ty::Shape(_) | Ty::Func(_) | Ty::Nil
        );
        let element = if may_be_flat {
            if_value(is_flat, flat, element, self.ir(ty), span)
        } else {
            element
        };
        let general = self.coerce(general, ty);
        Val {
            node: block_value(
                pre,
                if_value(in_range, element, general, self.ir(ty), span),
                span,
            ),
            ty,
        }
    }
}
