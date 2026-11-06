//! Constructors for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn table_constructor(
        &mut self,
        t: &ast::TableConstructor,
        span: Span,
    ) -> Result<Val> {
        self.table_constructor_with(t, None, span)
    }

    /// A constructor; with `meta`, the table is born with that
    /// metatable, which the caller has checked the constructor's shape
    /// has slots for.
    pub(super) fn table_constructor_with(
        &mut self,
        t: &ast::TableConstructor,
        meta: Option<&Expression>,
        span: Span,
    ) -> Result<Val> {
        let fields: Vec<&ast::Field> = t.fields().iter().collect();
        let positional: Vec<&Expression> = fields
            .iter()
            .filter_map(|f| match f {
                ast::Field::NoKey(e) => Some(e),
                _ => None,
            })
            .collect();
        // No positional values: the shared empty array part, made by
        // no one.
        let arr = if positional.is_empty() {
            None
        } else {
            Some(self.packed_list(&positional, span)?)
        };
        let table_t = self.ir(Ty::Table);
        // A constructor of a slotted shape lays its table out itself.
        if let Some(k) = types::constructor_shape(self.m.inferred, t)
            && self.m.layout(k).is_some()
        {
            return self.shaped_constructor(k, &fields, arr, positional.len(), meta, span);
        }
        debug_assert!(meta.is_none(), "a metatable at birth needs a slotted shape");
        let table = match arr {
            Some((arr, Some(Ty::Int))) => {
                call("zl_table_with_flat", vec![arr], table_t.clone(), span)
            }
            Some((arr, Some(_))) => {
                call("zl_table_with_flat_float", vec![arr], table_t.clone(), span)
            }
            Some((arr, None)) => call("zl_table_with_arr", vec![arr], table_t.clone(), span),
            None => call("zl_table_new", vec![], table_t.clone(), span),
        };
        let keyed: Vec<&ast::Field> = fields
            .iter()
            .copied()
            .filter(|f| !matches!(f, ast::Field::NoKey(_)))
            .collect();
        if keyed.is_empty() {
            return Ok(Val {
                node: table,
                ty: Ty::Table,
            });
        }
        let name = self.temp();
        let mut pre = vec![let_(name, table_t.clone(), table, span)];
        for f in keyed {
            let tb = var(name, table_t.clone(), span);
            match f {
                ast::Field::NameKey { key, value, .. } => {
                    let v = self.expr(value)?;
                    let k = Val {
                        node: str_lit(&ident(key), span),
                        ty: Ty::Str,
                    };
                    pre.push(self.raw_store(tb, k, v, span));
                }
                ast::Field::ExpressionKey { key, value, .. } => {
                    let k = self.expr(key)?;
                    let v = self.expr(value)?;
                    pre.push(self.raw_store(tb, k, v, span));
                }
                _ => {}
            }
        }
        Ok(Val {
            node: block_value(pre, var(name, table_t, span), span),
            ty: Ty::Table,
        })
    }

    /// A constructor whose tables have shape `k`, laid out with its
    /// slots: the header, then each constant-key field in its slot,
    /// present when its value is not nil. Fields under other keys are
    /// stored afterwards, as into a plain table. `meta`, evaluated
    /// after the fields, is the metatable the table is born with.
    pub(super) fn shaped_constructor(
        &mut self,
        k: ShapeId,
        fields: &[&ast::Field],
        arr: Option<(Node, Option<Ty>)>,
        positional: usize,
        meta: Option<&Expression>,
        span: Span,
    ) -> Result<Val> {
        let table_t = self.ir(Ty::Table);
        let i64_t = prim(PrimitiveType::I64);
        let layout = self.m.layout(k).expect("a slotted shape");
        let shape_ty = self.m.shape_ty(layout);
        let gid = layout.gid;
        let slot_of = |name: &str| layout.slots.iter().find(|s| s.name == name).cloned();
        // The values, in source order, kept apart from the slots they
        // go in.
        let mut pre = Vec::new();
        let mut slot_values: Vec<(Slot, Val)> = Vec::new();
        let mut other: Vec<(Val, Val)> = Vec::new();
        for f in fields {
            match f {
                ast::Field::NameKey { key, value, .. } => {
                    let v = self.expr(value)?;
                    let v = self.hold(v, &mut pre);
                    let name = ident(key);
                    match slot_of(&name) {
                        Some(slot) => slot_values.push((slot, v)),
                        None => other.push((
                            Val {
                                node: str_lit(&name, span),
                                ty: Ty::Str,
                            },
                            v,
                        )),
                    }
                }
                ast::Field::ExpressionKey { key, value, .. } => {
                    let kv = self.expr(key)?;
                    let kv = self.hold(kv, &mut pre);
                    let v = self.expr(value)?;
                    let v = self.hold(v, &mut pre);
                    match crate::scope::literal_string(key).and_then(|n| slot_of(&n)) {
                        Some(slot) => slot_values.push((slot, v)),
                        None => other.push((kv, v)),
                    }
                }
                _ => {}
            }
        }
        // A table the lowering has just made has no metatable to
        // protect: `setmetatable` of it with a class that holds neither
        // `__gc` nor `__mode` is the header's store; anything else goes
        // through the library, which checks it and tells the collector.
        let (meta, meta_checked) = match meta {
            Some(e) => {
                let m = self.expr(e)?;
                let m = self.hold(m, &mut pre);
                let inferred = self.m.inferred;
                match m.ty {
                    Ty::Shape(c)
                        if !inferred.class_may_hold(c, "__gc")
                            && !inferred.class_may_hold(c, "__mode") =>
                    {
                        (m.node, None)
                    }
                    _ => (null(table_t.clone(), span), Some(self.boxed(m))),
                }
            }
            None => (null(table_t.clone(), span), None),
        };
        // A key given twice: the last value stands, as in a plain table.
        let mut inits: Vec<TypedFieldInit> = Vec::new();
        let mut present: Node = int_lit(0, span);
        let mut filled: Vec<usize> = Vec::new();
        let layout = self.m.layout(k).expect("a slotted shape");
        for slot in &layout.slots {
            let value = slot_values
                .iter()
                .rev()
                .find(|(s, _)| s.bit == slot.bit)
                .map(|(_, v)| v.clone());
            if slot.kind == SlotKind::Number {
                let (bits, words) = self.number_slot_init(slot, value, &mut pre, span);
                present = binary(BinaryOp::BitOr, present, bits, i64_t.clone(), span);
                inits.extend(words);
                continue;
            }
            let stored = match value {
                Some(v) => {
                    filled.push(slot.bit);
                    let stored = self.coerce(v, slot.stored_ty());
                    let stored = self
                        .hold(
                            Val {
                                node: stored,
                                ty: slot.stored_ty(),
                            },
                            &mut pre,
                        )
                        .node;
                    let bit = match self.slot_nil(&stored, slot, span) {
                        Some(is_nil) => if_value(
                            is_nil,
                            int_lit(0, span),
                            int_lit(slot.mask(), span),
                            i64_t.clone(),
                            span,
                        ),
                        None => int_lit(slot.mask(), span),
                    };
                    present = binary(BinaryOp::BitOr, present, bit, i64_t.clone(), span);
                    stored
                }
                // An absent slot holds nothing: null for a table, never
                // a table made for it.
                None => match slot.kind {
                    SlotKind::Table => null(table_t.clone(), span),
                    _ => self.zero_of(slot.stored_ty(), span),
                },
            };
            match slot.kind {
                SlotKind::Scalar => {
                    let parts = NumberParts::of(&stored, span);
                    for (suffix, part) in [("t", parts.tag), ("i", parts.int), ("f", parts.float)] {
                        inits.push(TypedFieldInit {
                            name: intern(&format!("s{}{suffix}", slot.bit)),
                            value: Box::new(part),
                        });
                    }
                }
                _ => inits.push(TypedFieldInit {
                    name: intern(&format!("s{}", slot.bit)),
                    value: Box::new(stored),
                }),
            }
        }
        // The struct's initializers are plain reads: the presence word
        // and the array part are settled ahead of it.
        let present = self
            .hold(
                Val {
                    node: present,
                    ty: Ty::Int,
                },
                &mut pre,
            )
            .node;
        let arr = match arr {
            Some((arr, Some(Ty::Int))) => call("zl_arr_box_flat", vec![arr], Type::Any, span),
            Some((arr, Some(_))) => call("zl_arr_box_flat_float", vec![arr], Type::Any, span),
            Some((arr, None)) => call("zl_arr_box", vec![arr], Type::Any, span),
            // The shared empty one, made at state entry.
            None => var(intern(library::ARR_EMPTY), Type::Any, span),
        };
        let arr = self
            .hold(
                Val {
                    node: arr,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node;
        let header = [
            ("arr", arr),
            ("hash", nil(span)),
            ("meta", meta),
            ("high", int_lit(positional as i64, span)),
            ("shape", int_lit(gid, span)),
            ("present", present),
        ];
        let mut all: Vec<TypedFieldInit> = header
            .into_iter()
            .map(|(name, value)| TypedFieldInit {
                name: intern(name),
                value: Box::new(value),
            })
            .collect();
        all.extend(inits);
        let object = node(
            TypedExpression::Struct(TypedStructLiteral {
                name: intern(&layout.name),
                fields: all,
            }),
            shape_ty,
            span,
        );
        // The table as every holder sees it.
        let table = as_shape(object, table_t.clone(), span);
        let _ = filled;
        if other.is_empty() && meta_checked.is_none() {
            return Ok(Val {
                node: block_value(pre, table, span),
                ty: Ty::Shape(k),
            });
        }
        let name = self.temp();
        pre.push(let_(name, table_t.clone(), table, span));
        for (key, value) in other {
            let tb = var(name, table_t.clone(), span);
            pre.push(self.raw_store(tb, key, value, span));
        }
        if let Some(m) = meta_checked {
            let set = call(
                "zl_setmetatable",
                vec![var(name, table_t.clone(), span), m],
                table_t.clone(),
                span,
            );
            pre.push(self.guarded_stmt(set, &Described::NONE));
        }
        Ok(Val {
            node: block_value(pre, var(name, table_t, span), span),
            ty: Ty::Shape(k),
        })
    }

    /// `t[k] = v` without metamethods, on a table.
    pub(super) fn raw_store(&mut self, tb: Node, k: Val, v: Val, span: Span) -> St {
        let v = self.boxed(v);
        let unit = prim(PrimitiveType::Unit);
        let node = if let Some(kb) = self.constant_key(&k) {
            call("zl_rawset_key", vec![tb, kb, v], unit, span)
        } else {
            match k.ty {
                Ty::Int => call("zl_rawseti", vec![tb, k.node, v], unit, span),
                Ty::Str => call("zl_rawset_str", vec![tb, k.node, v], unit, span),
                _ => {
                    let k = self.boxed(k);
                    call("zl_rawset", vec![tb, k, v], unit, span)
                }
            }
        };
        self.guarded_stmt(node, &Described::NONE)
    }
}
