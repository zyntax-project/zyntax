//! Builtins for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A call to a library function: each argument as the
    /// implementation takes it.
    pub(super) fn builtin_call(
        &mut self,
        b: &Builtin,
        receiver: Option<Val>,
        args: &ast::FunctionArgs,
        span: Span,
    ) -> Result<Multi> {
        // `rawget(_G, "x")` and `rawset(_G, "x", v)` are the global.
        if b.lib.is_empty() && (b.name == "rawget" || b.name == "rawset") {
            let exprs = self.args_exprs(args);
            if let Some(Expression::Var(Var::Name(token))) = exprs.first() {
                let is_g = matches!(self.scopes().binding(token), Some(Binding::Global(g)) if Scopes::is_globals_name(g))
                    && !self.scopes().global_writes.contains_key(&ident(token))
                    && !self.scopes().dynamic_globals;
                if is_g {
                    let name = exprs.get(1).and_then(|e| crate::scope::literal_string(e));
                    let Some(name) = name else {
                        return unsupported("`_G` indexed by anything but a string literal", span);
                    };
                    if b.name == "rawget" {
                        let v = self.read_global(&name, span)?;
                        return Ok(Multi::Fixed(vec![v]));
                    }
                    let Some(value) = exprs.get(2) else {
                        return unsupported("`rawset(_G, name)` without a value", span);
                    };
                    let v = self.expr(value)?;
                    let st = self.write_global(&name, v, span)?;
                    return Ok(Multi::None(block_value(vec![st], nil(span), span)));
                }
            }
        }
        // `setmetatable` of a table a constructor is making: the table
        // is born with its metatable.
        if b.lib.is_empty()
            && b.name == "setmetatable"
            && receiver.is_none()
            && let [first, mt] = self.args_exprs(args).as_slice()
            && let Expression::TableConstructor(t) = first
            && let Some(k) = types::constructor_shape(self.m.inferred, t)
            && self.m.layout(k).is_some()
        {
            let v = self.table_constructor_with(t, Some(mt), span_of(first))?;
            return Ok(Multi::Fixed(vec![v]));
        }
        let is_method = receiver.is_some();
        // `error(v, level)` above level 1 positions at the caller: the
        // line this function was entered at.
        if b.lib.is_empty() && b.name == "error" {
            let exprs = self.args_exprs(args);
            let level = exprs.get(1).and_then(|e| match e {
                Expression::Number(n) => {
                    match crate::host::parse_numeral(n.token().to_string().trim()) {
                        crate::host::Numeral::Int(v) => Some(v),
                        _ => None,
                    }
                }
                _ => None,
            });
            if level.is_some_and(|l| l >= 2) {
                self.entry_line = true;
                let (mut pre, vals, _) = self.call_values(None, args, span)?;
                let v = vals.into_iter().next().map(|v| self.boxed(v));
                let v = v.unwrap_or_else(|| nil(span));
                let call = call(
                    "zl_error_at",
                    vec![v, var(intern(ENTRY_LINE), prim(PrimitiveType::I64), span)],
                    prim(PrimitiveType::Unit),
                    span,
                );
                let after = self.around_builtin(b, is_method, span, &mut pre);
                pre.push(expr_stmt(call));
                pre.extend(after);
                pre.push(self.pending_check(span));
                return Ok(Multi::None(block_value(pre, nil(span), span)));
            }
        }
        let (mut pre, vals, tail) = self.call_values(receiver, args, span)?;
        let mut vals: Vec<Val> = vals.into_iter().map(|v| self.hold(v, &mut pre)).collect();
        if tail.is_none()
            && let Some(m) = self.math_call(b, &mut vals, &mut pre, span)
        {
            return Ok(m);
        }
        if tail.is_none()
            && let Some(m) = self.simple_string_format(b, &mut vals, &mut pre, span)
        {
            return Ok(m);
        }
        if tail.is_none()
            && let Some(m) = self.typed_print(b, is_method, &mut vals, &mut pre, span)
        {
            return Ok(m);
        }
        // `tostring` of a string or a number is its text, made as `..`
        // makes it.
        if b.lib.is_empty()
            && b.name == "tostring"
            && tail.is_none()
            && vals.len() == 1
            && matches!(vals[0].ty, Ty::Str | Ty::Int | Ty::Float | Ty::Number)
        {
            let text = self.text_of(vals.remove(0));
            return Ok(Multi::Fixed(vec![Val {
                node: block_value(pre, text, span),
                ty: Ty::Str,
            }]));
        }
        // `table.sort` of a table with no metatable by a function the
        // types know: a sort of its own, calling the comparator
        // directly with the elements as their type.
        if b.lib == "table"
            && b.name == "sort"
            && tail.is_none()
            && vals.len() == 2
            && let Ty::Shape(k) = vals[0].ty
            && let Ty::Func(f) = vals[1].ty
            && self.m.plain_shape(k)
        {
            let helper = self.sort_helper(k, f, span);
            self.raise_callees.insert(f);
            let table_t = self.ir(Ty::Table);
            let bool_t = prim(PrimitiveType::Bool);
            let i64_t = prim(PrimitiveType::I64);
            let unit = prim(PrimitiveType::Unit);
            let comp = vals.pop().expect("the comparator");
            let t = vals.pop().expect("the table");
            let t_null = binary(
                BinaryOp::Eq,
                t.node.clone(),
                null(table_t, span),
                bool_t.clone(),
                span,
            );
            let comp_nil = binary(
                BinaryOp::Eq,
                comp.node.clone(),
                nil(span),
                bool_t.clone(),
                span,
            );
            let boxed_t = self.boxed(t.clone());
            let general = call(
                "zl_table_sort",
                vec![boxed_t, comp.node.clone()],
                unit.clone(),
                span,
            );
            let arr = self.temp();
            let typed = vec![
                let_(
                    arr,
                    self.m.anys(),
                    call("zl_arr_own", vec![t.node], self.m.anys(), span),
                    span,
                ),
                expr_stmt(call(
                    &helper,
                    vec![
                        var(arr, self.m.anys(), span),
                        int_lit(0, span),
                        binary(
                            BinaryOp::Sub,
                            list_len(var(arr, self.m.anys(), span), span),
                            int_lit(1, span),
                            i64_t.clone(),
                            span,
                        ),
                        var(intern(library::LINE), i64_t, span),
                        comp.node,
                    ],
                    unit,
                    span,
                )),
            ];
            // A nil table raises through the general path; a nil
            // comparator sorts by `<` there too.
            pre.push(if_(
                binary(BinaryOp::Or, t_null, comp_nil, bool_t, span),
                vec![expr_stmt(general)],
                Some(typed),
                span,
            ));
            pre.push(self.pending_check(span));
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        }
        // `setmetatable` of tables the types know: neither is boxed.
        if b.lib.is_empty()
            && b.name == "setmetatable"
            && tail.is_none()
            && vals.len() == 2
            && vals
                .iter()
                .all(|v| matches!(v.ty, Ty::Table | Ty::Shape(_)))
        {
            let table_t = self.ir(Ty::Table);
            let ty = vals[0].ty;
            let node = call(
                "zl_setmetatable_tables",
                vals.into_iter().map(|v| v.node).collect(),
                table_t,
                span,
            );
            let v = self.guard(Val {
                node,
                ty: Ty::Table,
            });
            return Ok(Multi::Fixed(vec![Val {
                node: block_value(pre, v.node, span),
                ty,
            }]));
        }
        // A tail of several values fills what follows through a list.
        let tail_list = match tail {
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
        let mut lowered = Vec::with_capacity(b.params.len());
        let mut any_bytes = false;
        let mut consumed = 0;
        for (i, p) in b.params.iter().enumerate() {
            if let Param::Rest = p {
                // Everything left, boxed, plus the tail.
                let mut items = Vec::new();
                for v in vals.drain(..) {
                    items.push(self.boxed(v));
                }
                let list = self.array_of(items, &mut pre, span);
                // The tail's values the parameters before took are
                // not the rest.
                let list = match tail_list {
                    Some(name) => {
                        let list_name = self.temp();
                        pre.push(let_(list_name, self.m.anys(), list, span));
                        let tail = var(name, self.m.anys(), span);
                        let rest = if consumed == 0 {
                            tail
                        } else {
                            call(
                                "zl_slice",
                                vec![
                                    tail.clone(),
                                    int_lit(consumed as i64, span),
                                    list_len(tail, span),
                                ],
                                self.m.anys(),
                                span,
                            )
                        };
                        pre.push(expr_stmt(call(
                            "zb_list_extend_any",
                            vec![var(list_name, self.m.anys(), span), rest],
                            prim(PrimitiveType::Unit),
                            span,
                        )));
                        var(list_name, self.m.anys(), span)
                    }
                    None => list,
                };
                lowered.push(list);
                consumed = usize::MAX;
                break;
            }
            let v = if !vals.is_empty() {
                Some(vals.remove(0))
            } else if let Some(name) = tail_list {
                consumed += 1;
                Some(Val {
                    node: call(
                        "zl_value_at",
                        vec![
                            var(name, self.m.anys(), span),
                            int_lit(consumed as i64, span),
                        ],
                        Type::Any,
                        span,
                    ),
                    ty: Ty::Any,
                })
            } else {
                None
            };
            // A method's receiver is not counted, and is not "bad
            // argument #0".
            let what = match (is_method, i) {
                (true, 0) => format!("calling '{}' on bad self", b.name),
                (true, i) => format!("bad argument #{i} to '{}'", b.name),
                (false, i) => format!("bad argument #{} to '{}'", i + 1, b.name),
            };
            let what = str_lit(&what, span);
            let arg = self.builtin_arg(p, v, what, span);
            any_bytes |= *p == Param::Bytes && arg.ty == Type::Any;
            lowered.push(arg);
        }
        // Arguments past the parameters run for their effects.
        if consumed != usize::MAX {
            for v in vals {
                pre.push(expr_stmt(v.node));
            }
        }
        let ret_ir = crate::library::stdlib::ret_type(b.ret, &self.m.types);
        // An argument's conversion can raise as well as the function;
        // it is checked before the call, which might otherwise take the
        // error as its own (`pcall`), each argument bound in order.
        if lowered.iter().any(|a| self.call_can_raise(a)) {
            for a in lowered.iter_mut() {
                let raises = self.call_can_raise(a);
                let name = self.temp();
                let ty = a.ty.clone();
                let node = std::mem::replace(a, var(name, ty.clone(), span));
                pre.push(let_(name, ty, node, span));
                if raises {
                    pre.push(self.pending_check(span));
                }
            }
        }
        let func = if any_bytes {
            crate::library::stdlib::any_form(b.func)
        } else {
            b.func.to_string()
        };
        let value = call(&func, lowered, ret_ir, span);
        let raises = self.call_can_raise(&value);
        let ty = match b.ret {
            Ret::Unit => Ty::Nil,
            Ret::Multi => Ty::Any,
            r => types::ret_ty(r),
        };
        let after = self.around_builtin(b, is_method, span, &mut pre);
        let v = if after.is_empty() {
            Val { node: value, ty }
        } else if b.ret == Ret::Unit {
            pre.push(expr_stmt(value));
            pre.extend(after);
            if raises {
                pre.push(self.pending_check(span));
            }
            return Ok(Multi::None(block_value(pre, nil(span), span)));
        } else {
            let held = self.hold(Val { node: value, ty }, &mut pre);
            pre.extend(after);
            held
        };
        let v = if raises { self.guard(v) } else { v };
        Ok(match b.ret {
            Ret::Unit => Multi::None(block_value(pre, v.node, span)),
            Ret::Multi => Multi::Dynamic(block_value(pre, v.node, span)),
            _ => Multi::Fixed(vec![Val {
                node: block_value(pre, v.node, span),
                ty: v.ty,
            }]),
        })
    }

    /// What a call of a library function does around it in a function
    /// that keeps a frame: one that may call the program back is a
    /// frame of its own, named as the reference names it; before one
    /// that reads the caller's locals, or calls something that may,
    /// the locals are spilled. Pushes what comes before the call onto
    /// `pre` and returns what comes after it.
    pub(super) fn around_builtin(
        &mut self,
        b: &Builtin,
        is_method: bool,
        span: Span,
        pre: &mut Vec<St>,
    ) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        // The frame is at the line the call starts on while it runs.
        pre.push(self.set_line(span));
        // The debug library's own functions are the level 0 it
        // describes, not a frame of the stack. `error` is the level an
        // error is raised at.
        let own_frame = (self.m.reentrant.contains(b.func) && b.lib != "debug")
            || (b.lib.is_empty() && b.name == "error");
        let reads_frame = b.lib == "debug" && matches!(b.name, "getlocal" | "setlocal");
        if !own_frame && !reads_frame {
            return Vec::new();
        }
        let global = if b.lib.is_empty() {
            b.name.to_string()
        } else {
            format!("{}.{}", b.lib, b.name)
        };
        let namewhat = if is_method {
            "method"
        } else if b.lib.is_empty() {
            "global"
        } else {
            "field"
        };
        let desc = Some(format!("{namewhat} '{}'", b.name));
        let Some(site) = self.debug_site_as(&desc, &global, span) else {
            return Vec::new();
        };
        let spill = self.spill(site, span, pre);
        let mut after = Vec::new();
        if own_frame {
            pre.push(expr_stmt(call(
                "zl_dbg_enter_c",
                vec![int_lit(site, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
            after.push(expr_stmt(call(
                "zl_dbg_leave",
                vec![],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        after.extend(self.after_call(spill, span));
        after
    }
}
