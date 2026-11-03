//! Dispatch for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A call through `callee`, a value the types say is one of
    /// `funcs`: each is called directly when the guard says it is the
    /// one, and anything else (nil, above all) is called through the
    /// value, which raises as Lua does. Every path yields the call's
    /// results the same way, typed as the call is. The arguments
    /// (`vals` and `tail`, with `pre` before them) run once.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch(
        &mut self,
        callee: Val,
        guard: Guard,
        funcs: &[FuncId],
        returns: Returns,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        span: Span,
    ) -> Multi {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        // The callee is read before the arguments run.
        let mut head = Vec::new();
        let callee = self.hold(callee, &mut head);
        let code = match guard {
            Guard::ByCode => Some(self.hold(
                Val {
                    node: call("zl_func_id", vec![callee.node.clone()], i64_t.clone(), span),
                    ty: Ty::Int,
                },
                &mut head,
            )),
            Guard::NotNil => None,
        };
        head.append(&mut pre);
        let arms: Vec<(Node, FuncId, Val)> = funcs
            .iter()
            .map(|f| {
                let is_f = match &code {
                    Some(code) => binary(
                        BinaryOp::Eq,
                        code.node.clone(),
                        int_lit(self.m.func_key(*f), span),
                        bool_t.clone(),
                        span,
                    ),
                    None => binary(
                        BinaryOp::Ne,
                        callee.node.clone(),
                        nil(span),
                        bool_t.clone(),
                        span,
                    ),
                };
                (is_f, *f, callee.clone())
            })
            .collect();
        self.dispatch_arms(callee, arms, returns, head, vals, tail, desc, span)
    }

    /// The arms of a dispatch around its fallback: `arms` are guards,
    /// in order, each with the function it calls and the record the
    /// call goes through; `callee` (held) is what the fallback calls.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_arms(
        &mut self,
        callee: Val,
        arms: Vec<(Node, FuncId, Val)>,
        returns: Returns,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        span: Span,
    ) -> Multi {
        let vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut pre)).collect();
        let tail = tail.map(|t| {
            self.hold(
                Val {
                    node: t,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node
        });
        // What every path yields: the results as one value.
        let shape = match &returns {
            Returns::Fixed(types) => Yield::Fixed(types.iter().map(|t| t.settled()).collect()),
            Returns::Dynamic | Returns::Unknown => Yield::Dynamic,
        };
        // The fallback first, then each function's arm around it.
        let boxed_callee = self.boxed(callee.clone());
        let fallback = self.value_call_vals(
            boxed_callee,
            Vec::new(),
            vals.clone(),
            tail.clone(),
            desc.clone(),
            false,
            span,
        );
        let mut value = self.yield_as(fallback, &shape, span);
        for (is_f, f, record) in arms.into_iter().rev() {
            let arm = self.direct_call_vals(
                f,
                Some(record),
                Vec::new(),
                vals.clone(),
                tail.clone(),
                &desc,
                span,
            );
            let then = self.yield_as(arm, &shape, span);
            let ty = then.ty.clone();
            value = if_value(is_f, then, value, ty, span);
        }
        self.yielded(block_value(pre, value, span), &shape, span)
    }

    /// The results of `multi` as the one value a dispatch arm yields:
    /// the value itself for one result or a dynamic one, a list of the
    /// boxed results for several, nil for none.
    pub(super) fn yield_as(&mut self, multi: Multi, shape: &Yield, span: Span) -> Node {
        match shape {
            Yield::Dynamic => match multi {
                Multi::Dynamic(node) => node,
                Multi::None(node) => block_value(vec![expr_stmt(node)], nil(span), span),
                Multi::Fixed(vals) => {
                    let mut pre = Vec::new();
                    let mut items = Vec::with_capacity(vals.len());
                    for v in vals {
                        items.push(self.boxed(v));
                    }
                    let list = self.array_of(items, &mut pre, span);
                    block_value(pre, call("zb_box_tuple", vec![list], Type::Any, span), span)
                }
            },
            Yield::Fixed(types) => {
                // The results, each as its type.
                let mut pre = Vec::new();
                let vals: Vec<Val> = match multi {
                    Multi::Fixed(vals) => {
                        let mut vals = vals.into_iter();
                        let mut out = Vec::with_capacity(types.len());
                        for ty in types {
                            let v = vals.next().unwrap_or_else(|| self.nil_val(span));
                            let v = Val {
                                node: self.coerce(v, *ty),
                                ty: *ty,
                            };
                            out.push(self.hold(v, &mut pre));
                        }
                        for v in vals {
                            pre.push(expr_stmt(v.node));
                        }
                        out
                    }
                    Multi::None(node) => {
                        pre.push(expr_stmt(node));
                        types
                            .iter()
                            .map(|ty| Val {
                                node: self.zero_of(*ty, span),
                                ty: *ty,
                            })
                            .collect()
                    }
                    Multi::Dynamic(node) => {
                        let list = self.temp();
                        pre.push(let_(
                            list,
                            self.m.anys(),
                            call("zl_values", vec![node], self.m.anys(), span),
                            span,
                        ));
                        types
                            .iter()
                            .enumerate()
                            .map(|(i, ty)| {
                                let element = Val {
                                    node: call(
                                        "zl_value_at",
                                        vec![
                                            var(list, self.m.anys(), span),
                                            int_lit(i as i64 + 1, span),
                                        ],
                                        Type::Any,
                                        span,
                                    ),
                                    ty: Ty::Any,
                                };
                                Val {
                                    node: self.coerce(element, *ty),
                                    ty: *ty,
                                }
                            })
                            .collect()
                    }
                };
                match types.len() {
                    0 => block_value(pre, nil(span), span),
                    1 => block_value(pre, vals.into_iter().next().unwrap().node, span),
                    _ => {
                        let mut items = Vec::with_capacity(vals.len());
                        for v in vals {
                            items.push(self.boxed(v));
                        }
                        let list = self.array_of(items, &mut pre, span);
                        block_value(pre, list, span)
                    }
                }
            }
        }
    }

    /// The one value a dispatch yielded, back as the call's results.
    pub(super) fn yielded(&mut self, value: Node, shape: &Yield, span: Span) -> Multi {
        match shape {
            Yield::Dynamic => Multi::Dynamic(value),
            Yield::Fixed(types) => match types.len() {
                0 => Multi::None(value),
                1 => Multi::Fixed(vec![Val {
                    node: value,
                    ty: types[0],
                }]),
                _ => {
                    let name = self.temp();
                    let mut pre = vec![let_(name, self.m.anys(), value, span)];
                    let mut vals = Vec::with_capacity(types.len());
                    for (i, ty) in types.iter().enumerate() {
                        let element = Val {
                            node: index(
                                var(name, self.m.anys(), span),
                                int_lit(i as i64, span),
                                Type::Any,
                                span,
                            ),
                            ty: Ty::Any,
                        };
                        let read = self.coerce(element, *ty);
                        let read = if i == 0 {
                            block_value(std::mem::take(&mut pre), read, span)
                        } else {
                            read
                        };
                        vals.push(Val {
                            node: read,
                            ty: *ty,
                        });
                    }
                    Multi::Fixed(vals)
                }
            },
        }
    }

    /// Statements run before several values.
    pub(super) fn prefixed(&mut self, pre: Vec<St>, multi: Multi, span: Span) -> Multi {
        if pre.is_empty() {
            return multi;
        }
        match multi {
            Multi::Dynamic(node) => Multi::Dynamic(block_value(pre, node, span)),
            Multi::None(node) => Multi::None(block_value(pre, node, span)),
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    Multi::Fixed(vec![Val {
                        node: block_value(pre, nil(span), span),
                        ty: Ty::Nil,
                    }])
                } else {
                    let first = vals.remove(0);
                    let mut out = vec![Val {
                        node: block_value(pre, first.node, span),
                        ty: first.ty,
                    }];
                    out.extend(vals);
                    Multi::Fixed(out)
                }
            }
        }
    }
}
