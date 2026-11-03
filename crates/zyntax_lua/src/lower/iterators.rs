//! Iterators for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn generic_for(
        &mut self,
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let names: Vec<VarId> = f
            .names()
            .iter()
            .map(|n| self.scopes().declared(n))
            .collect();
        let exprs: Vec<&Expression> = f.expressions().iter().collect();
        // `pairs(t)`, `ipairs(t)` and `next, t` walk the table itself.
        if let [Expression::FunctionCall(c)] = exprs.as_slice() {
            let suffixes: Vec<&Suffix> = c.suffixes().collect();
            if let Some(b) = self.typer().builtin_callee(c.prefix(), &suffixes)
                && b.lib.is_empty()
                && (b.name == "pairs" || b.name == "ipairs")
            {
                let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.last() else {
                    unreachable!("a builtin callee ends in a call")
                };
                let arg_exprs = self.args_exprs(args);
                if let (1, None) = (arg_exprs.len(), self.literal_arg(args, span)?) {
                    let t = self.expr(arg_exprs[0])?;
                    return if b.name == "ipairs" {
                        self.ipairs_loop(&names, t, f, span, out)
                    } else {
                        self.pairs_loop(&names, t, f, span, out)
                    };
                }
            }
        }
        if let [Expression::Var(Var::Name(token)), table] = exprs.as_slice()
            && let Some(Binding::Global(name)) = self.scopes().binding(token)
            && name == "next"
            && types::builtin_named(self.scopes(), "next").is_some()
        {
            let t = self.expr(table)?;
            return self.pairs_loop(&names, t, f, span, out);
        }
        if !self.debug_locals()
            && let Some(captures) = self.typer().gmatch_walk(&exprs)
        {
            return self.gmatch_loop(&names, &exprs, &captures, f, span, out);
        }
        // The general protocol: `f, s, control` and a closing value,
        // then `f(s, control)` until its first value is nil. A closing
        // value comes from a fourth expression that is not nil, or from
        // the last expression's values when fewer are written.
        let listed = self.expr_list(&exprs)?;
        let (_, fixed, tail) = &listed;
        let closes = match fixed.get(3) {
            Some(v) => v.ty != Ty::Nil,
            None => tail.is_some(),
        };
        let (pre, vals) = self.adjust(listed, if closes { 4 } else { 3 }, span);
        out.extend(pre);
        let mut it = vals.into_iter();
        let (fv, sv, cv) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
        let fname = self.temp();
        let sname = self.temp();
        let cname = self.temp();
        let fb = self.boxed(fv);
        let sb = self.boxed(sv);
        let cb = self.boxed(cv);
        out.push(let_(fname, Type::Any, fb, span));
        out.push(let_(sname, Type::Any, sb, span));
        out.push(let_(cname, Type::Any, cb, span));
        // The closing value is checked once, before the first call, and
        // closed however the loop is left.
        let closing = match it.next() {
            Some(value) => {
                let name = self.temp();
                let value = self.boxed(value);
                out.push(let_(name, Type::Any, value, span));
                let check = call(
                    "zl_closable",
                    vec![var(name, Type::Any, span), str_lit("(for state)", span)],
                    prim(PrimitiveType::Unit),
                    span,
                );
                let st = self.guarded_stmt(check, &Described::NONE);
                out.push(st);
                self.tbc.push((self.depth + 1, Closable::Loop(name)));
                Some(name)
            }
            None => None,
        };
        let vals_name = self.temp();
        let step = call(
            "zl_call_2",
            vec![
                var(fname, Type::Any, span),
                var(sname, Type::Any, span),
                var(cname, Type::Any, span),
            ],
            Type::Any,
            span,
        );
        let step = self
            .guard(Val {
                node: step,
                ty: Ty::Any,
            })
            .node;
        let mut body = vec![let_(
            vals_name,
            self.m.anys(),
            call("zl_values", vec![step], self.m.anys(), span),
            span,
        )];
        let first = call(
            "zl_value_at",
            vec![var(vals_name, self.m.anys(), span), int_lit(1, span)],
            Type::Any,
            span,
        );
        body.push(assign(var(cname, Type::Any, span), first, span));
        body.push(if_(
            binary(
                BinaryOp::Eq,
                var(cname, Type::Any, span),
                nil(span),
                prim(PrimitiveType::Bool),
                span,
            ),
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        for (i, v) in names.iter().enumerate() {
            let value = if i == 0 {
                var(cname, Type::Any, span)
            } else {
                call(
                    "zl_value_at",
                    vec![
                        var(vals_name, self.m.anys(), span),
                        int_lit(i as i64 + 1, span),
                    ],
                    Type::Any,
                    span,
                )
            };
            body.push(self.declare_var(
                *v,
                Val {
                    node: value,
                    ty: Ty::Any,
                },
                span,
            ));
        }
        let states = vec![
            var(fname, Type::Any, span),
            var(sname, Type::Any, span),
            var(cname, Type::Any, span),
            closing.map_or_else(|| nil(span), |name| var(name, Type::Any, span)),
        ];
        body.extend(self.for_body(f.block(), states, &names, span_of(f.end_token()))?);
        body.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        out.push(while_(bool_lit(true, span), body, span));
        if closing.is_some() {
            self.closes_checked(self.depth + 1, true, span_of(f.end_token()), out);
            self.tbc.pop();
        }
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// `for i, v in ipairs(t)`: `t[1], t[2], …` until nil.
    pub(super) fn ipairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let block = f.block();
        let i64_t = prim(PrimitiveType::I64);
        let tname = self.temp();
        let counter = self.temp();
        let value = self.temp();
        let t_ir = self.ir(t.ty);
        let t_ty = t.ty;
        out.push(let_(tname, t_ir.clone(), t.node, span));
        out.push(let_(counter, i64_t.clone(), int_lit(1, span), span));
        let read = self.index_read(
            Val {
                node: var(tname, t_ir, span),
                ty: t_ty,
            },
            Val {
                node: var(counter, i64_t.clone(), span),
                ty: Ty::Int,
            },
            None,
            span,
        );
        // The element as the read types it; the loop ends at nil.
        let read_ty = read.ty;
        let read_ir = self.ir(read_ty);
        let mut body = vec![let_(value, read_ir.clone(), read.node, span)];
        let at_end = self.is_nil_val(Val {
            node: var(value, read_ir.clone(), span),
            ty: read_ty,
        });
        body.push(if_(
            at_end,
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        if let Some(k) = names.first() {
            body.push(self.declare_var(
                *k,
                Val {
                    node: var(counter, i64_t.clone(), span),
                    ty: Ty::Int,
                },
                span,
            ));
        }
        if let Some(v) = names.get(1) {
            body.push(self.declare_var(
                *v,
                Val {
                    node: var(value, read_ir, span),
                    ty: read_ty,
                },
                span,
            ));
        }
        for v in names.iter().skip(2) {
            let n = self.nil_val(span);
            body.push(self.declare_var(*v, n, span));
        }
        // What `ipairs(t)` would have given: its iterator, `t`, and the
        // index as the control.
        let states = if self.debug_locals() {
            let t = self.boxed(Val {
                node: var(tname, self.ir(t_ty), span),
                ty: t_ty,
            });
            let iter = self.temp();
            let triple = call(
                "zl_values",
                vec![call("zl_ipairs", vec![t.clone()], Type::Any, span)],
                self.m.anys(),
                span,
            );
            out.push(let_(
                iter,
                Type::Any,
                call(
                    "zl_value_at",
                    vec![triple, int_lit(1, span)],
                    Type::Any,
                    span,
                ),
                span,
            ));
            let index = self.boxed(Val {
                node: var(counter, i64_t.clone(), span),
                ty: Ty::Int,
            });
            vec![var(iter, Type::Any, span), t, index, nil(span)]
        } else {
            vec![nil(span); 4]
        };
        let mut inner = self.for_body(block, states, names, span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let increment = assign(
            var(counter, i64_t.clone(), span),
            binary(
                BinaryOp::Add,
                var(counter, i64_t.clone(), span),
                int_lit(1, span),
                i64_t,
                span,
            ),
            span,
        );
        self.push_loop_with_increment(body, inner, increment, span, out);
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// `for ... in string.gmatch(s, p)`, `s` a string and `p` a literal
    /// pattern: each match found in place by `zl_gmatch_next` and its
    /// captures read from the matcher straight after, a string or a
    /// position each (`captures`), as the iterator would give them.
    pub(super) fn gmatch_loop(
        &mut self,
        names: &[VarId],
        exprs: &[&Expression],
        captures: &[bool],
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let [Expression::FunctionCall(c)] = exprs else {
            unreachable!("a gmatch walk is one call")
        };
        let suffixes: Vec<&Suffix> = c.suffixes().collect();
        let Some(Suffix::Call(ast::Call::AnonymousCall(args))) = suffixes.last() else {
            unreachable!("a gmatch walk ends in its call")
        };
        let arg_exprs = self.args_exprs(args);
        let subject = self.expr(arg_exprs[0])?;
        let pattern = self.expr(arg_exprs[1])?;
        let i64_t = prim(PrimitiveType::I64);
        let str_t = self.ir(Ty::Str);
        let [s, p, n, from, last, end] = std::array::from_fn(|_| self.temp());
        let get = |name, ty: &Type| var(name, ty.clone(), span);
        out.push(let_(s, str_t.clone(), subject.node, span));
        out.push(let_(p, str_t.clone(), pattern.node, span));
        out.push(let_(
            n,
            i64_t.clone(),
            call("zb_str_len", vec![get(s, &str_t)], i64_t.clone(), span),
            span,
        ));
        out.push(let_(from, i64_t.clone(), int_lit(0, span), span));
        out.push(let_(last, i64_t.clone(), int_lit(-1, span), span));
        let next = self.guard(Val {
            node: call(
                "zl_gmatch_next",
                vec![
                    get(s, &str_t),
                    get(p, &str_t),
                    get(from, &i64_t),
                    get(n, &i64_t),
                    get(last, &i64_t),
                ],
                i64_t.clone(),
                span,
            ),
            ty: Ty::Int,
        });
        let mut body = vec![let_(end, i64_t.clone(), next.node, span)];
        body.push(if_(
            binary(
                BinaryOp::Lt,
                get(end, &i64_t),
                int_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            vec![stmt(TypedStatement::Break(None), span)],
            None,
            span,
        ));
        body.push(assign(get(from, &i64_t), get(end, &i64_t), span));
        body.push(assign(get(last, &i64_t), get(end, &i64_t), span));
        for (j, v) in names.iter().enumerate() {
            let value = match captures.get(j) {
                Some(false) => Val {
                    node: call(
                        "zl_pat_cap_str",
                        vec![get(s, &str_t), int_lit(j as i64, span)],
                        str_t.clone(),
                        span,
                    ),
                    ty: Ty::Str,
                },
                Some(true) => Val {
                    node: call(
                        "zl_pat_cap_pos",
                        vec![int_lit(j as i64, span)],
                        i64_t.clone(),
                        span,
                    ),
                    ty: Ty::Int,
                },
                None => self.nil_val(span),
            };
            body.push(self.declare_var(*v, value, span));
        }
        let mut inner =
            self.for_body(f.block(), vec![nil(span); 4], names, span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        body.extend(inner);
        out.push(while_(bool_lit(true, span), body, span));
        let end_line = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end_line), span));
        Ok(())
    }
}
