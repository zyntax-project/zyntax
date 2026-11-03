//! Blocks for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The chunk's outermost block as segments of at most
    /// `SEGMENT_STATEMENTS` statements each, the final `return` in the
    /// last.
    pub(super) fn segments(&mut self, block: &Block) -> Result<Vec<Vec<St>>> {
        self.depth = 1;
        let mut segments = Vec::new();
        let mut out = Vec::new();
        let entry = self.present.clone();
        for s in block.stmts() {
            if let Stmt::Label(_) = s {
                self.present = entry.clone();
            }
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_stmt_line(Some(s), span_of(s), at, &mut out);
            if out.len() >= crate::scope::SEGMENT_STATEMENTS {
                segments.push(std::mem::take(&mut out));
            }
        }
        if let Some(last) = block.last_stmt() {
            let span = span_of(last);
            let at = out.len();
            self.line_needed = false;
            match last {
                ast::LastStmt::Return(r) => {
                    let exprs: Vec<&Expression> = r.returns().iter().collect();
                    self.return_stmt(&exprs, span, &mut out)?;
                }
                _ => return unsupported("this statement", span),
            }
            self.record_stmt_line(None, span, at, &mut out);
        } else {
            let span = span_of(block);
            self.closes_checked(
                1,
                true,
                Span::new(span.end.saturating_sub(1), span.end),
                &mut out,
            );
        }
        if !out.is_empty() || segments.is_empty() {
            segments.push(out);
        }
        Ok(segments)
    }

    pub(super) fn block(&mut self, block: &Block) -> Result<Vec<St>> {
        self.depth += 1;
        let live = self.live.len();
        let mut out = Vec::new();
        // A `goto` reaches a label from anywhere in the block, past
        // stores that may not have run.
        let entry = self.present.clone();
        for s in block.stmts() {
            if let Stmt::Label(_) = s {
                self.present = entry.clone();
            }
            let at = out.len();
            self.line_needed = false;
            self.stmt(s, &mut out)?;
            self.record_stmt_line(Some(s), span_of(s), at, &mut out);
        }
        match block.last_stmt() {
            Some(last) => {
                let span = span_of(last);
                let at = out.len();
                self.line_needed = false;
                match last {
                    ast::LastStmt::Break(_) => {
                        // The loop's body and everything within it is left.
                        let (loop_depth, end) =
                            self.loop_depths.last().copied().unwrap_or((1, span));
                        self.closes_checked(loop_depth, false, end, &mut out);
                        out.push(stmt(TypedStatement::Break(None), span));
                    }
                    ast::LastStmt::Return(r) => {
                        let exprs: Vec<&Expression> = r.returns().iter().collect();
                        self.return_stmt(&exprs, span, &mut out)?;
                    }
                    _ => return unsupported("this statement", span),
                }
                self.record_stmt_line(None, span, at, &mut out);
            }
            None => {
                // A function's body closes at its `end`, any other
                // block at its last token.
                let end = match self.end_span.filter(|_| self.depth == 1) {
                    Some(end) => end,
                    None => {
                        let span = span_of(block);
                        Span::new(span.end.saturating_sub(1), span.end)
                    }
                };
                self.closes_checked(self.depth, false, end, &mut out);
            }
        }
        let depth = self.depth;
        self.tbc
            .retain(|(d, c)| *d < depth || (*d == depth && matches!(c, Closable::Loop(_))));
        self.depth -= 1;
        self.live.truncate(live);
        Ok(out)
    }

    /// What leaving to `depth` closes, innermost first: everything
    /// entered at `depth` or deeper, a loop's closing value at `depth`
    /// itself only when `loops` (the jump leaves that loop).
    pub(super) fn closing(&self, depth: usize, loops: bool) -> Vec<(usize, Closable)> {
        self.tbc
            .iter()
            .rev()
            .filter(|(d, c)| {
                *d > depth || (*d == depth && (loops || matches!(c, Closable::Var(_))))
            })
            .copied()
            .collect()
    }

    /// Close everything entered at `depth` or deeper, innermost first.
    /// Each handler sees the error pending, if any. The entries stay in
    /// scope for the paths that go on.
    pub(super) fn closes_from(&mut self, depth: usize, span: Span, out: &mut Vec<St>) {
        let entries = self.closing(depth, true);
        self.emit_closes(&entries, span, out);
    }

    pub(super) fn emit_closes(
        &mut self,
        entries: &[(usize, Closable)],
        span: Span,
        out: &mut Vec<St>,
    ) {
        if !entries.is_empty() {
            // A handler may raise, and its error is this function's.
            self.raised = true;
        }
        for (_, c) in entries {
            let value = match *c {
                Closable::Var(v) => {
                    let value = self.read_var(v, span);
                    self.boxed(value)
                }
                Closable::Loop(name) => var(name, Type::Any, span),
            };
            out.push(expr_stmt(call(
                "zl_close",
                vec![value],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
    }

    /// The closes of a jump or a block's end, on the line of `at`. When
    /// a handler raised, the error then leaves at once, closing only
    /// what is still open.
    pub(super) fn closes_checked(
        &mut self,
        depth: usize,
        loops: bool,
        at: Span,
        out: &mut Vec<St>,
    ) {
        let entries = self.closing(depth, loops);
        if entries.is_empty() {
            return;
        }
        out.push(self.set_line(at));
        self.emit_closes(&entries, at, out);
        let scope = self.tbc.clone();
        self.tbc.retain(|e| !entries.contains(e));
        let check = self.pending_check(at);
        self.tbc = scope;
        out.push(check);
    }

    /// The line a statement starts, stored ahead of it. A function
    /// that keeps a frame stores every statement's line, the one its
    /// code runs on (a function definition's is its `end`'s), and
    /// tells the hooks; a block's own line runs nothing.
    pub(super) fn record_stmt_line(
        &mut self,
        s: Option<&Stmt>,
        span: Span,
        at: usize,
        out: &mut Vec<St>,
    ) {
        if !self.frames {
            self.record_line(span, at, out);
            return;
        }
        let line = match s {
            Some(Stmt::Do(_) | Stmt::Repeat(_)) => {
                self.line_needed = false;
                return;
            }
            Some(Stmt::LocalFunction(f)) => self.m.line_of(span_of(f.body().end_token())),
            Some(Stmt::FunctionDeclaration(f)) => self.m.line_of(span_of(f.body().end_token())),
            _ => self.m.line_of(span),
        };
        self.line_needed = false;
        let statements = self.debug_line(line, span);
        out.splice(at..at, statements);
    }

    /// A statement that checks for an error stores its line first, for
    /// the position the error's message carries.
    pub(super) fn record_line(&mut self, span: Span, at: usize, out: &mut Vec<St>) {
        if !self.line_needed {
            return;
        }
        self.line_needed = false;
        let st = self.set_line(span);
        out.insert(at, st);
    }

    pub(super) fn return_stmt(
        &mut self,
        exprs: &[&Expression],
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        // `return f(...)` is a tail call, unless something is to be
        // closed after it.
        let tail = match exprs {
            _ if !self.tbc.is_empty() => None,
            [e @ Expression::FunctionCall(_)] => Some(span_of(*e)),
            [e @ Expression::Var(Var::Expression(v))]
                if matches!(v.suffixes().last(), Some(Suffix::Call(_))) =>
            {
                Some(span_of(*e))
            }
            _ => None,
        };
        let outer_tail = std::mem::replace(&mut self.tail_span, tail);
        let values = self.return_values(exprs, span, out);
        self.tail_span = outer_tail;
        values?;
        if self.tbc.is_empty() && !self.counts_depth && !self.frames {
            return Ok(());
        }
        // The values are computed before anything is closed.
        let Some(last) = out.pop() else {
            return Ok(());
        };
        let TypedStatement::Return(value) = last.node else {
            out.push(last);
            return Ok(());
        };
        let value = value.map(|value| {
            let name = self.temp();
            let ty = value.ty.clone();
            out.push(let_(name, ty.clone(), *value, span));
            var(name, ty, span)
        });
        self.closes_from(1, span, out);
        if self.counts_depth {
            out.push(depth_step(-1, span));
        }
        out.extend(self.debug_call("zl_dbg_leave", Vec::new(), span));
        out.push(ret(value, span));
        Ok(())
    }

    pub(super) fn return_values(
        &mut self,
        exprs: &[&Expression],
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let returns = self.returns.clone();
        match returns {
            Returns::Fixed(types) if types.is_empty() => {
                for e in exprs {
                    let v = self.expr(e)?;
                    out.push(expr_stmt(v.node));
                }
                if self.func == CHUNK && self.scopes().split_chunk {
                    let flag = intern(RETURNED);
                    self.m.declare_module_var(flag, Ty::Bool);
                    out.push(assign(
                        var(flag, prim(PrimitiveType::Bool), span),
                        bool_lit(true, span),
                        span,
                    ));
                }
                out.push(ret(None, span));
            }
            Returns::Fixed(types) if types.len() == 1 => {
                let (pre, mut vals) = self.adjusted(exprs, 1, span)?;
                out.extend(pre);
                let v = vals.pop().unwrap();
                let value = self.coerce(v, types[0].settled());
                out.push(ret(Some(value), span));
            }
            Returns::Fixed(types) => {
                let (pre, vals) = self.adjusted(exprs, types.len(), span)?;
                out.extend(pre);
                let mut items = Vec::with_capacity(vals.len());
                for v in vals {
                    items.push(self.boxed(v));
                }
                let mut pre = Vec::new();
                let list = self.array_of(items, &mut pre, span);
                out.extend(pre);
                out.push(ret(Some(list), span));
            }
            Returns::Dynamic | Returns::Unknown => {
                let (pre, vals, tail) = self.expr_list(exprs)?;
                out.extend(pre);
                let value = match (vals.len(), tail) {
                    (1, None) => {
                        let v = vals.into_iter().next().unwrap();
                        self.boxed(v)
                    }
                    (0, Some(tail)) => tail,
                    (_, tail) => {
                        let mut items = Vec::with_capacity(vals.len());
                        for v in vals {
                            items.push(self.boxed(v));
                        }
                        let mut pre = Vec::new();
                        let list = self.array_of(items, &mut pre, span);
                        out.extend(pre);
                        match tail {
                            None => call("zl_pack", vec![list], Type::Any, span),
                            Some(tail) => {
                                let name = self.temp();
                                let pre = vec![
                                    let_(name, self.m.anys(), list, span),
                                    expr_stmt(call(
                                        "zl_append_values",
                                        vec![var(name, self.m.anys(), span), tail],
                                        prim(PrimitiveType::Unit),
                                        span,
                                    )),
                                ];
                                block_value(
                                    pre,
                                    call(
                                        "zl_pack",
                                        vec![var(name, self.m.anys(), span)],
                                        Type::Any,
                                        span,
                                    ),
                                    span,
                                )
                            }
                        }
                    }
                };
                out.push(ret(Some(value), span));
            }
        }
        Ok(())
    }
}
