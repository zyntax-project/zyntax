//! Value calls for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A call through a function value, which `desc` names.
    pub(super) fn value_call(
        &mut self,
        callee: Val,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        desc: Desc,
        span: Span,
    ) -> Result<Multi> {
        // A value the types know the function of calls it directly,
        // once it is seen not to be nil.
        if let Ty::Func(f) = callee.ty {
            let returns = self.m.sig(f).returns.clone();
            let (pre, vals, tail) = self.call_values(receiver, args, span)?;
            return Ok(self.dispatch(
                callee,
                Guard::NotNil,
                &[f],
                returns,
                pre,
                vals,
                tail,
                desc,
                span,
            ));
        }
        // A value known to be nil cannot be called; the error names
        // what it was, as Lua's does.
        if callee.ty == Ty::Nil {
            let (mut pre, _, _) = self.call_values(receiver, args, span)?;
            let what = match desc {
                Some(d) => format!("attempt to call a nil value ({d})"),
                None => "attempt to call a nil value".to_string(),
            };
            let error = call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit(&what, span)],
                prim(PrimitiveType::Unit),
                span,
            );
            pre.push(self.guarded_stmt(error, &Described::NONE));
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        }
        let f = self.boxed(callee);
        let method = receiver.is_some();
        let (pre, vals, tail) = self.call_values(receiver, args, span)?;
        Ok(self.value_call_vals(f, pre, vals, tail, desc, method, span))
    }

    /// [`Self::value_call`] with the callee boxed and the arguments
    /// evaluated. A `method` call enters through the method-site
    /// entries, which tell a library function its first argument is
    /// `self`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn value_call_vals(
        &mut self,
        f: Node,
        mut pre: Vec<St>,
        vals: Vec<Val>,
        tail: Option<Node>,
        desc: Desc,
        method: bool,
        span: Span,
    ) -> Multi {
        let entry = if method {
            "zl_apply_method"
        } else {
            "zl_apply"
        };
        // A call site that keeps the debug library's record evaluates
        // everything ahead of storing its number.
        let frames = self.frames;
        let f = if pre.is_empty() && !frames {
            f
        } else {
            self.hold(
                Val {
                    node: f,
                    ty: Ty::Any,
                },
                &mut pre,
            )
            .node
        };
        let vals: Vec<Val> = if frames {
            vals.into_iter().map(|v| self.hold(v, &mut pre)).collect()
        } else {
            vals
        };
        let tail = match tail {
            Some(t) if frames => Some(
                self.hold(
                    Val {
                        node: t,
                        ty: Ty::Any,
                    },
                    &mut pre,
                )
                .node,
            ),
            t => t,
        };
        let descs = Described::callee(desc.clone());
        let called = if tail.is_none() && vals.len() <= zyntax_builtins::functions::MAX_CALL_ARITY {
            let mut lowered = vec![f];
            let n = vals.len();
            for v in vals {
                lowered.push(self.boxed(v));
            }
            call(&format!("{entry}_{n}"), lowered, Type::Any, span)
        } else {
            let mut items = Vec::with_capacity(vals.len());
            for v in vals {
                items.push(self.boxed(v));
            }
            let list = self.array_of(items, &mut pre, span);
            let list = match tail {
                None => list,
                Some(tail) => {
                    let name = self.temp();
                    pre.push(let_(name, self.m.anys(), list, span));
                    pre.push(expr_stmt(call(
                        "zl_append_values",
                        vec![var(name, self.m.anys(), span), tail],
                        prim(PrimitiveType::Unit),
                        span,
                    )));
                    var(name, self.m.anys(), span)
                }
            };
            call(&format!("{entry}_packed"), vec![f, list], Type::Any, span)
        };
        let spill = self.before_call(&desc, span, &mut pre);
        let mut after = self.after_call(spill, span);
        // A callee of the library's enters no frame to take the site.
        if frames {
            after.push(assign(
                var(
                    intern(library::debug::DBG_SITE),
                    prim(PrimitiveType::I64),
                    span,
                ),
                int_lit(0, span),
                span,
            ));
        }
        let called = if after.is_empty() {
            called
        } else {
            let held = self.hold(
                Val {
                    node: called,
                    ty: Ty::Any,
                },
                &mut pre,
            );
            pre.extend(after);
            held.node
        };
        let v = self.guard_described(
            Val {
                node: called,
                ty: Ty::Any,
            },
            &descs,
        );
        Multi::Dynamic(block_value(pre, v.node, span))
    }

    /// `obj:name(args)`: `obj.name(obj, args)` with `obj` evaluated
    /// once; `desc` names `obj`.
    pub(super) fn method_call(
        &mut self,
        obj: Val,
        mc: &ast::MethodCall,
        desc: Desc,
        exact: Option<ShapeId>,
        span: Span,
    ) -> Result<Multi> {
        let name = ident(mc.name());
        let mut pre = Vec::new();
        let obj = self.hold(obj, &mut pre);
        // A string's methods are the string library's.
        if obj.ty == Ty::Str
            && let Some(b) = types::builtin_member(self.scopes(), "string", &name)
        {
            let multi = self.builtin_call(b, Some(obj), mc.args(), span)?;
            return Ok(self.prefixed(pre, multi, span));
        }
        let key = Val {
            node: str_lit(&name, span),
            ty: Ty::Str,
        };
        let method = Some(format!("method '{name}'"));
        // A method the types resolve to known functions is called
        // directly, whichever of them the receiver's class holds: told
        // by which lookup ends it when every table on its way has
        // slots, else by the code of the value found.
        if let Ty::Shape(k) = obj.ty
            && let Some(targets) = self.typer().method_targets(k, &name)
        {
            let funcs: Vec<FuncId> = targets
                .iter()
                .filter_map(|t| match t {
                    Ty::Func(f) => Some(*f),
                    _ => None,
                })
                .collect();
            if !funcs.is_empty() {
                let returns = self.typer().method_returns(Ty::Shape(k), &name).settled();
                if let Some(multi) = self.which_call(
                    obj.clone(),
                    k,
                    &name,
                    CallKind::Method,
                    exact,
                    returns.clone(),
                    mc.args(),
                    desc.clone(),
                    span,
                )? {
                    return Ok(self.prefixed(pre, multi, span));
                }
                let (arg_pre, vals, tail) = self.call_values(Some(obj.clone()), mc.args(), span)?;
                let callee = self.index_read(obj, key, desc, span);
                let multi = self.dispatch(
                    callee,
                    Guard::ByCode,
                    &funcs,
                    returns,
                    arg_pre,
                    vals,
                    tail,
                    method,
                    span,
                );
                return Ok(self.prefixed(pre, multi, span));
            }
        }
        let callee = self.index_read(obj.clone(), key, desc, span);
        let multi = self.value_call(callee, Some(obj), mc.args(), method, span)?;
        Ok(self.prefixed(pre, multi, span))
    }
}
