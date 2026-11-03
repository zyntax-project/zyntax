//! Direct calls for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn args_exprs<'e>(&self, args: &'e ast::FunctionArgs) -> Vec<&'e Expression> {
        match args {
            ast::FunctionArgs::Parentheses { arguments, .. } => arguments.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// The single argument a call written `f "x"` or `f {…}` passes.
    pub(super) fn literal_arg(
        &mut self,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<Option<Val>> {
        Ok(match args {
            ast::FunctionArgs::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Some(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            ast::FunctionArgs::TableConstructor(t) => Some(self.table_constructor(t, span)?),
            _ => None,
        })
    }

    /// The values a call's arguments supply: `receiver` first for a
    /// method call.
    pub(super) fn call_values(
        &mut self,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut vals = Vec::new();
        if let Some(r) = receiver {
            vals.push(r);
        }
        if let Some(v) = self.literal_arg(args, span)? {
            vals.push(v);
            return Ok((Vec::new(), vals, None));
        }
        let exprs = self.args_exprs(args);
        let (mut pre, more, tail) = self.expr_list(&exprs)?;
        if !pre.is_empty() {
            // The receiver runs before the arguments.
            let held = std::mem::take(&mut vals);
            let mut first = Vec::new();
            for v in held {
                let h = self.hold(v, &mut first);
                vals.push(h);
            }
            first.append(&mut pre);
            pre = first;
        }
        vals.extend(more);
        Ok((pre, vals, tail))
    }

    /// A call to a known function: its typed entry, the arguments
    /// adjusted to its parameters. `record` is the function's value,
    /// when the call went through one, which holds its captures;
    /// `receiver` is the object of a method call, passed first.
    pub(super) fn direct_call(
        &mut self,
        f: FuncId,
        record: Option<Val>,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Multi> {
        // The function's value is read before its arguments run.
        let mut pre = Vec::new();
        let record = record.map(|r| self.hold(r, &mut pre));
        let (mut arg_pre, vals, tail) = self.call_values(receiver, args, span)?;
        pre.append(&mut arg_pre);
        Ok(self.direct_call_vals(f, record, pre, vals, tail, &desc, span))
    }

    /// [`Self::direct_call`] with the arguments evaluated: `vals`, and
    /// `tail` when the last supplies several values.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn direct_call_vals(
        &mut self,
        f: FuncId,
        record: Option<Val>,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: &Desc,
        span: Span,
    ) -> Multi {
        self.note_argument_fields(f, vals.first());
        let info = self.scopes().func(f);
        let sig = self.m.sig(f);
        let n = info.params.len();
        let is_vararg = info.is_vararg;
        let has_env = self.m.takes_env(f);
        // A tail of several values is read through a list.
        let tail_name = match tail {
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                Some(name)
            }
            None => None,
        };
        let mut lowered: Vec<Node> = Vec::new();
        if has_env {
            // A nested known function's record holds its captures; the
            // typed entry takes it as `env`. The record is the value
            // the call went through, else it lives in the function's
            // variable.
            let record = match record {
                Some(r) => r,
                None => match self
                    .scopes()
                    .local_functions
                    .iter()
                    .find(|(_, id)| **id == f)
                {
                    Some((v, _)) => self.read_var(*v, span),
                    None => self.function_value(f, span),
                },
            };
            lowered.push(call(
                "zb_unbox_list_raw_any",
                vec![record.node],
                self.m.anys(),
                span,
            ));
        }
        let mut fixed = vals.into_iter();
        let mut from_tail: i64 = 0;
        for i in 0..n {
            let param_ty = sig.params.get(i).copied().unwrap_or(Ty::Any).settled();
            let v = match fixed.next() {
                Some(v) => v,
                None => match tail_name {
                    Some(t) => {
                        from_tail += 1;
                        Val {
                            node: call(
                                "zl_value_at",
                                vec![var(t, self.m.anys(), span), int_lit(from_tail, span)],
                                Type::Any,
                                span,
                            ),
                            ty: Ty::Any,
                        }
                    }
                    None => self.nil_val(span),
                },
            };
            let v = self.hold(v, &mut pre);
            lowered.push(self.coerce(v, param_ty));
        }
        // The extras: the variadic list, or run for their effects.
        let extras: Vec<Val> = fixed.collect();
        if is_vararg {
            let mut items = Vec::with_capacity(extras.len());
            for v in extras {
                // Evaluated ahead of the call site's own work.
                let v = if self.frames {
                    self.hold(v, &mut pre)
                } else {
                    v
                };
                items.push(self.boxed(v));
            }
            let list = self.array_of(items, &mut pre, span);
            match tail_name {
                Some(t) => {
                    let list_name = self.temp();
                    pre.push(let_(list_name, self.m.anys(), list, span));
                    let rest = call(
                        "zl_slice",
                        vec![
                            var(t, self.m.anys(), span),
                            int_lit(from_tail, span),
                            list_len(var(t, self.m.anys(), span), span),
                        ],
                        self.m.anys(),
                        span,
                    );
                    pre.push(expr_stmt(call(
                        "zb_list_extend_any",
                        vec![var(list_name, self.m.anys(), span), rest],
                        prim(PrimitiveType::Unit),
                        span,
                    )));
                    lowered.push(var(list_name, self.m.anys(), span));
                }
                None => lowered.push(list),
            }
        } else {
            for v in extras {
                pre.push(expr_stmt(v.node));
            }
        }
        let spill = self.before_call(desc, span, &mut pre);
        let value = call(
            &self.m.entry_name(f),
            lowered,
            self.m.return_ir(&sig.returns),
            span,
        );
        let after = self.after_call(spill, span);
        let raises = self.m.raises(f);
        if raises {
            self.raise_callees.insert(f);
        }
        self.call_result(value, &sig.returns, pre, after, raises, span)
    }

    /// The values a typed entry returned, checked for an error when the
    /// callee may raise.
    /// `after` runs once the call returned, ahead of its check.
    pub(super) fn call_result(
        &mut self,
        value: Node,
        returns: &Returns,
        pre: Vec<St>,
        after: Vec<St>,
        raises: bool,
        span: Span,
    ) -> Multi {
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                let mut pre = pre;
                pre.push(expr_stmt(value));
                pre.extend(after);
                if raises {
                    pre.push(self.pending_check(span));
                }
                Multi::None(block_value(pre, nil(span), span))
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let ty = types[0].settled();
                let mut pre = pre;
                let value = if raises || !after.is_empty() {
                    let held = self.hold(Val { node: value, ty }, &mut pre);
                    pre.extend(after);
                    if raises {
                        pre.push(self.pending_check(span));
                    }
                    held.node
                } else {
                    value
                };
                Multi::Fixed(vec![Val {
                    node: block_value(pre, value, span),
                    ty,
                }])
            }
            Returns::Fixed(types) => {
                // Several values: a list, each read out as its type.
                let name = self.temp();
                let mut pre = pre;
                pre.push(let_(name, self.m.anys(), value, span));
                pre.extend(after);
                if raises {
                    pre.push(self.pending_check(span));
                }
                let mut vals = Vec::with_capacity(types.len());
                for (i, ty) in types.iter().enumerate() {
                    let ty = ty.settled();
                    let element = index(
                        var(name, self.m.anys(), span),
                        int_lit(i as i64, span),
                        Type::Any,
                        span,
                    );
                    let read = self.coerce(
                        Val {
                            node: element,
                            ty: Ty::Any,
                        },
                        ty,
                    );
                    let read = if i == 0 {
                        block_value(std::mem::take(&mut pre), read, span)
                    } else {
                        read
                    };
                    vals.push(Val { node: read, ty });
                }
                Multi::Fixed(vals)
            }
            Returns::Dynamic | Returns::Unknown => {
                let mut pre = pre;
                let value = if after.is_empty() {
                    value
                } else {
                    let held = self.hold(
                        Val {
                            node: value,
                            ty: Ty::Any,
                        },
                        &mut pre,
                    );
                    pre.extend(after);
                    held.node
                };
                let value = if raises {
                    self.guard(Val {
                        node: value,
                        ty: Ty::Any,
                    })
                    .node
                } else {
                    value
                };
                Multi::Dynamic(block_value(pre, value, span))
            }
        }
    }
}
