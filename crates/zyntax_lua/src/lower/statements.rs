//! Statements for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn stmt(&mut self, s: &Stmt, out: &mut Vec<St>) -> Result<()> {
        // What a nested block stores may not have run once it is left.
        let nested = matches!(
            s,
            Stmt::Do(_)
                | Stmt::If(_)
                | Stmt::While(_)
                | Stmt::Repeat(_)
                | Stmt::NumericFor(_)
                | Stmt::GenericFor(_)
        );
        let before = nested.then(|| self.present.clone());
        self.stmt_of(s, out)?;
        if let Some(before) = before {
            self.present = before;
        }
        // The locals a declaration makes are in scope from the next
        // statement on.
        match s {
            Stmt::LocalAssignment(l) => {
                for name in l.names() {
                    let v = self.scopes().declared(name);
                    if !self.scopes().var(v).folded {
                        self.live.push(Live::Var(v));
                    }
                }
            }
            Stmt::LocalFunction(f) => {
                let v = self.scopes().declared(f.name());
                if !self.scopes().pruned.contains(&v) {
                    self.live.push(Live::Var(v));
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn stmt_of(&mut self, s: &Stmt, out: &mut Vec<St>) -> Result<()> {
        let span = span_of(s);
        match s {
            Stmt::LocalAssignment(l) => {
                self.note_fresh_locals(l);
                let names: Vec<&TokenReference> = l.names().iter().collect();
                let exprs: Vec<&Expression> = l.expressions().iter().collect();
                let (pre, vals) = if exprs.is_empty() {
                    (
                        Vec::new(),
                        names.iter().map(|_| self.nil_val(span)).collect(),
                    )
                } else {
                    self.adjusted(&exprs, names.len(), span)?
                };
                out.extend(pre);
                for (name, v) in names.iter().zip(vals) {
                    let id = self.scopes().declared(name);
                    out.push(self.declare_var(id, v, span));
                    // A `<close>` variable must hold something closable,
                    // and is closed when its block is left.
                    if self.scopes().var(id).attribute.as_deref() == Some("close") {
                        let value = self.read_var(id, span);
                        let value = self.boxed(value);
                        let check = call(
                            "zl_closable",
                            vec![value, str_lit(&ident(name), span)],
                            prim(PrimitiveType::Unit),
                            span,
                        );
                        let st = self.guarded_stmt(check, &Described::NONE);
                        out.push(st);
                        self.tbc.push((self.depth, Closable::Var(id)));
                    }
                }
            }
            Stmt::Assignment(a) => {
                let targets: Vec<&Var> = a.variables().iter().collect();
                let exprs: Vec<&Expression> = a.expressions().iter().collect();
                if targets.len() == 1 && exprs.len() == 1 {
                    // The common case: no temporaries.
                    let v = self.expr(exprs[0])?;
                    let st = self.assign_target(targets[0], v, span)?;
                    out.push(st);
                    self.note_stores(&targets, &exprs);
                    return Ok(());
                }
                // Every target's table and key, then every right side,
                // run before any target is stored.
                let mut pre = Vec::new();
                let mut prepared = Vec::with_capacity(targets.len());
                for target in &targets {
                    prepared.push(self.prepare_target(target, &mut pre, span)?);
                }
                let (mut values_pre, vals) = self.adjusted(&exprs, targets.len(), span)?;
                // A value that is a variable is read now: a store may
                // assign it before the value is used.
                let vals: Vec<Val> = vals
                    .into_iter()
                    .map(|v| self.snapshot(v, &mut values_pre))
                    .collect();
                out.extend(pre);
                out.extend(values_pre);
                for (target, v) in prepared.into_iter().zip(vals) {
                    let st = self.store_prepared(target, v, span)?;
                    out.push(st);
                }
                self.note_stores(&targets, &exprs);
            }
            Stmt::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                match multi {
                    Multi::Fixed(vals) => {
                        for v in vals {
                            out.push(expr_stmt(v.node));
                        }
                    }
                    Multi::Dynamic(node) | Multi::None(node) => out.push(expr_stmt(node)),
                }
            }
            Stmt::Do(d) => {
                let body = self.block(d.block())?;
                out.push(stmt(
                    TypedStatement::Block(TypedBlock {
                        statements: body,
                        span,
                    }),
                    span,
                ));
            }
            Stmt::If(i) => {
                let st = self.if_stmt(i, span)?;
                out.push(st);
            }
            Stmt::While(w) => {
                let cond = self.expr(w.condition())?;
                let cond = self.truthy(cond);
                let mut body = self.loop_body(w.block(), span_of(w.end_token()))?;
                body.extend(self.debug_loop_back(span_of(w.while_token()), true, span));
                out.push(while_(cond, body, span));
            }
            Stmt::Repeat(r) => {
                // `repeat body until c` is a loop leaving once `c` holds;
                // the condition sees the body's locals.
                // A `break` closes from the condition's last token.
                let until = span_of(r.until());
                let until = Span::new(until.end.saturating_sub(1), until.end);
                let mut body = self.loop_body(r.block(), until)?;
                if self.frames {
                    let line = self.m.line_of(span_of(r.until_token()));
                    body.extend(self.debug_line(line, span));
                }
                let cond = self.expr(r.until())?;
                let cond = self.truthy(cond);
                body.push(if_(
                    cond,
                    vec![stmt(TypedStatement::Break(None), span)],
                    None,
                    span,
                ));
                body.extend(self.debug_call("zl_dbg_back", Vec::new(), span));
                out.push(while_(bool_lit(true, span), body, span));
            }
            Stmt::NumericFor(f) => self.numeric_for(f, span, out)?,
            Stmt::GenericFor(f) => self.generic_for(f, span, out)?,
            Stmt::LocalFunction(f) => {
                let v = self.scopes().declared(f.name());
                if self.scopes().pruned.contains(&v) {
                    return Ok(());
                }
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                // The record is made once, here, and lives in the
                // variable; a direct call reads its captures from it. A
                // function capturing its own variable needs the cell
                // before the record.
                if matches!(self.storage_of(v), Storage::Cell(..)) {
                    let placeholder = self.nil_val(span);
                    out.push(self.declare_var(v, placeholder, span));
                    let record = self.function_value(id, span);
                    out.push(self.write_var(v, record, span));
                } else {
                    let record = self.function_value(id, span);
                    out.push(self.declare_var(v, record, span));
                }
            }
            Stmt::FunctionDeclaration(f) => {
                let id = self.scopes().function_of(f.body());
                let is_method = f.name().method_name().is_some();
                // The function is reached only through the field it is
                // stored in, so its body runs with the field there.
                self.note_declaration(f);
                self.lower_function(id, f.body(), is_method)?;
                let names: Vec<&TokenReference> = f.name().names().iter().collect();
                if names.len() == 1 && !is_method {
                    let token = names[0];
                    let binding = self
                        .scopes()
                        .binding(token)
                        .cloned()
                        .unwrap_or_else(|| Binding::Global(ident(token)));
                    if let Binding::Global(name) = &binding
                        && self.scopes().known_global_function(name) == Some(id)
                    {
                        // The typed entry is the function.
                        return Ok(());
                    }
                    let record = self.function_value(id, span);
                    let st = match binding {
                        Binding::Local(v) | Binding::Upvalue(v) => self.write_var(v, record, span),
                        Binding::Global(name) => self.write_global(&name, record, span)?,
                        Binding::Field(v, name, upvalue) => {
                            self.write_field(v, &name, upvalue, record, span)
                        }
                    };
                    out.push(st);
                    return Ok(());
                }
                // `function a.b.c()` / `function a.b:m()`: index down to
                // the holder, then store.
                let mut obj = self.read_name(names[0])?;
                let mut desc = self.describe_name(names[0]);
                let mut pre = Vec::new();
                for name in &names[1..names.len() - if is_method { 0 } else { 1 }] {
                    let key = Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    };
                    obj = self.index_read(obj, key, desc, span);
                    desc = Some(format!("field '{}'", ident(name)));
                    obj = self.hold(obj, &mut pre);
                }
                let last = match f.name().method_name() {
                    Some(m) => ident(m),
                    None => ident(names[names.len() - 1]),
                };
                let record = self.function_value(id, span);
                let key = Val {
                    node: str_lit(&last, span),
                    ty: Ty::Str,
                };
                out.extend(pre);
                let st = self.index_write(obj, key, record, desc, span);
                out.push(st);
            }
            Stmt::Goto(g) => {
                let Some(&id) = self
                    .scopes()
                    .gotos
                    .get(&crate::scope::pos_of(g.goto_token()))
                else {
                    return Err(Error::Syntax {
                        message: format!("no visible label '{}' for goto", ident(g.label_name())),
                        span: (span.start, span.end),
                    });
                };
                let target_depth = self.scopes().label_depths[&id];
                self.closes_checked(target_depth + 1, true, span, out);
                out.push(stmt(TypedStatement::Goto(label_name(id)), span));
            }
            Stmt::Label(l) => {
                let id = self.scopes().labels[&crate::scope::pos_of(l.name())];
                out.push(stmt(TypedStatement::Label(label_name(id)), span));
            }
            _ => return unsupported("this statement", span),
        }
        Ok(())
    }

    /// Store into an assignment target.
    /// A target of a multiple assignment with its table and key
    /// evaluated ahead of the values, as the reference does.
    pub(super) fn prepare_target<'e>(
        &mut self,
        target: &'e Var,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Result<Prepared<'e>> {
        let Var::Expression(ve) = target else {
            return Ok(Prepared::Whole(target));
        };
        let suffixes: Vec<&Suffix> = ve.suffixes().collect();
        if self.typer().global_member(ve.prefix(), &suffixes).is_some() {
            return Ok(Prepared::Whole(target));
        }
        let Some((last, init)) = suffixes.split_last() else {
            return unsupported("this assignment target", span);
        };
        let obj_multi = self.suffixed(ve.prefix(), init, span)?;
        let obj = self.first_of(obj_multi, span);
        let obj = self.snapshot(obj, pre);
        let desc = self.describe_chain(ve.prefix(), init);
        let key = match last {
            Suffix::Index(ast::Index::Dot { name, .. }) => Val {
                node: str_lit(&ident(name), span),
                ty: Ty::Str,
            },
            Suffix::Index(ast::Index::Brackets { expression, .. }) => self.expr(expression)?,
            _ => return unsupported("this assignment target", span),
        };
        let key = self.snapshot(key, pre);
        Ok(Prepared::Index(Box::new((obj, key, desc))))
    }

    pub(super) fn store_prepared(
        &mut self,
        target: Prepared<'_>,
        v: Val,
        span: Span,
    ) -> Result<St> {
        match target {
            Prepared::Whole(target) => self.assign_target(target, v, span),
            Prepared::Index(place) => {
                let (obj, key, desc) = *place;
                Ok(self.index_write(obj, key, v, desc, span))
            }
        }
    }

    pub(super) fn assign_target(&mut self, target: &Var, v: Val, span: Span) -> Result<St> {
        match target {
            Var::Name(token) => {
                let binding = self
                    .scopes()
                    .binding(token)
                    .cloned()
                    .unwrap_or_else(|| Binding::Global(ident(token)));
                match binding {
                    Binding::Local(id) | Binding::Upvalue(id) => {
                        if self.scopes().var(id).attribute.as_deref() == Some("const") {
                            return Err(Error::Syntax {
                                message: format!(
                                    "attempt to assign to const variable '{}'",
                                    ident(token)
                                ),
                                span: (span.start, span.end),
                            });
                        }
                        Ok(self.write_var(id, v, span))
                    }
                    Binding::Global(name) => self.write_global(&name, v, span),
                    Binding::Field(id, name, upvalue) => {
                        Ok(self.write_field(id, &name, upvalue, v, span))
                    }
                }
            }
            Var::Expression(ve) => {
                let suffixes: Vec<&Suffix> = ve.suffixes().collect();
                if let Some(name) = self.typer().global_member(ve.prefix(), &suffixes) {
                    return self.write_global(&name, v, span);
                }
                let Some((last, init)) = suffixes.split_last() else {
                    return unsupported("this assignment target", span);
                };
                let obj_multi = self.suffixed(ve.prefix(), init, span)?;
                let obj = self.first_of(obj_multi, span);
                let desc = self.describe_chain(ve.prefix(), init);
                let key = match last {
                    Suffix::Index(ast::Index::Dot { name, .. }) => Val {
                        node: str_lit(&ident(name), span),
                        ty: Ty::Str,
                    },
                    Suffix::Index(ast::Index::Brackets { expression, .. }) => {
                        self.expr(expression)?
                    }
                    _ => return unsupported("this assignment target", span),
                };
                Ok(self.index_write(obj, key, v, desc, span))
            }
            _ => unsupported("this assignment target", span),
        }
    }

    pub(super) fn if_stmt(&mut self, i: &ast::If, span: Span) -> Result<St> {
        let cond = self.expr(i.condition())?;
        let cond = self.truthy(cond);
        let then = self.block(i.block())?;
        // `elseif` chains nest as else-ifs.
        let mut els: Option<Vec<St>> = i.else_block().map(|b| self.block(b)).transpose()?;
        if let Some(elseifs) = i.else_if() {
            for e in elseifs.iter().rev() {
                let c = self.expr(e.condition())?;
                let c = self.truthy(c);
                let body = self.block(e.block())?;
                els = Some(vec![if_(c, body, els, span_of(e))]);
            }
        }
        Ok(if_(cond, then, els, span))
    }
}
