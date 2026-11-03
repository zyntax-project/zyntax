//! Method calls for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// `obj:name(args)`, or `obj.name(args)` by `kind`, for a receiver
    /// of shape `k` whose lookups of `name` all end in slots holding
    /// known functions or in `__index` functions giving one: the
    /// receiver checked
    /// not nil, then the `Which` finder says which end the lookup
    /// takes, and that end's function is called directly. Everything
    /// the call goes through (the end, a closure's record, what an
    /// `__index` function gives) is found before the arguments run,
    /// as Lua binds the callee first. A lookup that ends nowhere
    /// raises. None when the lookups are not all of that kind; `desc`
    /// names `obj`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn which_call(
        &mut self,
        obj: Val,
        k: ShapeId,
        name: &str,
        kind: CallKind,
        exact: Option<ShapeId>,
        returns: Returns,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Option<Multi>> {
        // A lookup decided here needs no finder.
        let folded = self
            .folded_class_lookups(k, name, exact)
            .and_then(|folded| self.fold_plan(k, name, folded));
        let (ends, dispatch) = match folded {
            Some((ends, plan)) => (ends, Dispatch::Folded(plan)),
            None => {
                let Some(which) = self.finder(k, name, FinderMode::Which, span) else {
                    return Ok(None);
                };
                (which.ends.clone(), Dispatch::Finder(which.helper))
            }
        };
        let direct = ends.iter().all(|(_, end)| match end {
            End::Slot(ty) => matches!(ty, Ty::Func(_)),
            End::Handler { ty, .. } => kind == CallKind::Method && matches!(ty, Ty::Func(_)),
        });
        if !direct {
            return Ok(None);
        }
        let takes_place = ends.iter().any(|(_, end)| match *end {
            End::Slot(Ty::Func(f)) => self.m.takes_env(f),
            End::Slot(_) => false,
            End::Handler { .. } => true,
        });
        let place = match &dispatch {
            Dispatch::Finder(_) if takes_place => {
                match self.finder(k, name, FinderMode::Where, span) {
                    Some(place) => Some(place),
                    None => return Ok(None),
                }
            }
            _ => None,
        };
        let table_t = self.ir(Ty::Table);
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut head = Vec::new();
        let obj = self.hold(obj, &mut head);
        // Exact receivers originate in successful fresh constructor calls.
        if exact.is_none() {
            // Indexing a nil receiver raises before anything is looked up.
            let mut raise = vec![
                self.set_line(span),
                expr_stmt(call("zl_index_nil", vec![], Type::Any, span)),
            ];
            raise.extend(self.leave_described(span, &Described::operand(desc)));
            head.push(if_(
                binary(
                    BinaryOp::Eq,
                    obj.node.clone(),
                    null(table_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                raise,
                None,
                span,
            ));
        }
        // Every table takes the one end: nothing is picked at run time.
        let sure = matches!(&dispatch, Dispatch::Folded(plan) if plan.arms.is_none());
        let (w, place) = match dispatch {
            Dispatch::Finder(helper) => {
                let w = self.hold(
                    Val {
                        node: call(&helper, vec![obj.node.clone()], i64_t.clone(), span),
                        ty: Ty::Int,
                    },
                    &mut head,
                );
                let place = place.map(|place| {
                    self.hold(
                        Val {
                            node: call(
                                &place.helper,
                                vec![obj.node.clone()],
                                table_t.clone(),
                                span,
                            ),
                            ty: Ty::Table,
                        },
                        &mut head,
                    )
                });
                (w, place)
            }
            Dispatch::Folded(plan) => {
                let meta = (plan.via_meta && (plan.arms.is_some() || takes_place)).then(|| {
                    self.hold(
                        Val {
                            node: field(obj.node.clone(), "meta", table_t.clone(), span),
                            ty: Ty::Table,
                        },
                        &mut head,
                    )
                });
                let w = match &plan.arms {
                    None => Val {
                        node: int_lit(0, span),
                        ty: Ty::Int,
                    },
                    // The end by the metatable's shape; none without one.
                    Some(arms) => {
                        let m = meta.as_ref().expect("a dispatch by metatable reads it");
                        let shape = field(m.node.clone(), "shape", i64_t.clone(), span);
                        let mut chosen = int_lit(-1, span);
                        for (gid, at) in arms.iter().rev() {
                            let is = binary(
                                BinaryOp::Eq,
                                shape.clone(),
                                int_lit(*gid, span),
                                bool_t.clone(),
                                span,
                            );
                            chosen = if_value(is, int_lit(*at, span), chosen, i64_t.clone(), span);
                        }
                        let no_meta = binary(
                            BinaryOp::Eq,
                            m.node.clone(),
                            null(table_t.clone(), span),
                            bool_t.clone(),
                            span,
                        );
                        self.hold(
                            Val {
                                node: if_value(
                                    no_meta,
                                    int_lit(-1, span),
                                    chosen,
                                    i64_t.clone(),
                                    span,
                                ),
                                ty: Ty::Int,
                            },
                            &mut head,
                        )
                    }
                };
                let place = takes_place.then(|| meta.unwrap_or_else(|| obj.clone()));
                (w, place)
            }
        };
        let is_end = |i: usize| {
            binary(
                BinaryOp::Eq,
                w.node.clone(),
                int_lit(i as i64, span),
                bool_t.clone(),
                span,
            )
        };
        // What each end calls through: a closure's record read from its
        // slot, or what the `__index` function gives for the name.
        let mut through: Vec<Option<Val>> = Vec::with_capacity(ends.len());
        for (i, (shape, end)) in ends.iter().enumerate() {
            let found = match (*end, &place) {
                (End::Slot(Ty::Func(f)), Some(place)) if self.m.takes_env(f) => {
                    let (layout, slot) = self
                        .slot_of(*shape, name)
                        .expect("a finder's slot end has a slot");
                    let record = self.slot_read(&place.node, layout, &slot, span);
                    Some(Val {
                        node: if sure {
                            record.node
                        } else {
                            if_value(is_end(i), record.node, nil(span), Type::Any, span)
                        },
                        ty: Ty::Any,
                    })
                }
                (End::Handler { class, f, ty }, Some(place)) => {
                    let (layout, slot) = self
                        .slot_of(class, "__index")
                        .expect("a finder's handler end has a slot");
                    let meta = field(place.node.clone(), "meta", table_t.clone(), span);
                    let record = self.slot_read(&meta, layout, &slot, span);
                    let args = vec![
                        Val {
                            node: place.node.clone(),
                            ty: Ty::Shape(*shape),
                        },
                        Val {
                            node: str_lit(name, span),
                            ty: Ty::Str,
                        },
                    ];
                    let multi = self.direct_call_vals(
                        f,
                        Some(record),
                        Vec::new(),
                        args,
                        None,
                        &Some("metamethod 'index'".to_string()),
                        span,
                    );
                    let ty = ty.settled();
                    let first = self.first_of(multi, span);
                    let first = self.coerce(first, ty);
                    let unread = self.zero_of(ty, span);
                    Some(Val {
                        node: if_value(is_end(i), first, unread, self.ir(ty), span),
                        ty,
                    })
                }
                _ => None,
            };
            through.push(found.map(|v| self.hold(v, &mut head)));
        }
        // Then the arguments, each held for the arm that takes it.
        let receiver = (kind == CallKind::Method).then(|| obj.clone());
        let (mut arg_pre, vals, tail) = self.call_values(receiver, args, span)?;
        head.append(&mut arg_pre);
        let vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut head)).collect();
        let tail = tail.map(|t| {
            self.hold(
                Val {
                    node: t,
                    ty: Ty::Any,
                },
                &mut head,
            )
            .node
        });
        let shape = match &returns {
            Returns::Fixed(types) => Yield::Fixed(types.iter().map(|t| t.settled()).collect()),
            Returns::Dynamic | Returns::Unknown => Yield::Dynamic,
        };
        let mut value = if sure {
            None
        } else {
            let raise = self.call_nil(name, kind, span);
            Some(self.yield_as(Multi::None(raise), &shape, span))
        };
        let callee_desc = Some(match kind {
            CallKind::Method => format!("method '{name}'"),
            CallKind::Field => format!("field '{name}'"),
        });
        for (i, (_, end)) in ends.iter().enumerate().rev() {
            let arm = match *end {
                End::Slot(Ty::Func(f)) => {
                    let multi = self.direct_call_vals(
                        f,
                        through[i].clone(),
                        Vec::new(),
                        vals.clone(),
                        tail.clone(),
                        &callee_desc,
                        span,
                    );
                    self.yield_as(multi, &shape, span)
                }
                End::Handler {
                    ty: Ty::Func(g), ..
                } => {
                    let callee = through[i].clone().expect("a handler end's value is held");
                    self.handler_arm(callee, g, name, &shape, &vals, &tail, &callee_desc, span)
                }
                _ => unreachable!("a which call's ends hold functions"),
            };
            value = Some(match value {
                Some(rest) => {
                    let ty = arm.ty.clone();
                    if_value(is_end(i), arm, rest, ty, span)
                }
                None => arm,
            });
        }
        let value = value.expect("a which call has an end");
        Ok(Some(self.yielded(
            block_value(head, value, span),
            &shape,
            span,
        )))
    }

    /// The arm of a method call whose lookup ends in an `__index`
    /// function, which gave `callee`, known to be the function `g` or
    /// nil: `g` called directly, or nil called, which raises.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn handler_arm(
        &mut self,
        callee: Val,
        g: FuncId,
        name: &str,
        shape: &Yield,
        vals: &[Val],
        tail: &Option<Node>,
        desc: &Desc,
        span: Span,
    ) -> Node {
        let multi = self.direct_call_vals(
            g,
            Some(callee.clone()),
            Vec::new(),
            vals.to_vec(),
            tail.clone(),
            desc,
            span,
        );
        let then = self.yield_as(multi, shape, span);
        let raise = self.call_nil(name, CallKind::Method, span);
        let otherwise = self.yield_as(Multi::None(raise), shape, span);
        let ty = then.ty.clone();
        let bound = binary(
            BinaryOp::Ne,
            callee.node,
            nil(span),
            prim(PrimitiveType::Bool),
            span,
        );
        if_value(bound, then, otherwise, ty, span)
    }

    /// Calling nil, found for `name`: the error, raised and left
    /// through.
    pub(super) fn call_nil(&mut self, name: &str, kind: CallKind, span: Span) -> Node {
        let mut raise = vec![
            self.set_line(span),
            expr_stmt(call(
                "zl_raise_call_nil",
                vec![str_lit(name, span), int_lit(kind as i64, span)],
                prim(PrimitiveType::Unit),
                span,
            )),
        ];
        let leave = self.leave_described(span, &Described::NONE);
        raise.push(if_(bool_lit(true, span), leave, None, span));
        block_value(raise, nil(span), span)
    }
}
