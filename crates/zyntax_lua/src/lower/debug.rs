//! Debug for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// `zl_line = <line>`, then the line and count hooks when a hook
    /// is set: a statement starts.
    pub(super) fn debug_line(&mut self, line: i64, span: Span) -> Vec<St> {
        let i64_t = prim(PrimitiveType::I64);
        self.lines.insert(line & ((1i64 << library::LINE_BITS) - 1));
        vec![
            assign(
                var(intern(library::LINE), i64_t.clone(), span),
                int_lit(line, span),
                span,
            ),
            self.if_hooked(
                vec![expr_stmt(call(
                    "zl_dbg_line",
                    vec![int_lit(line, span)],
                    prim(PrimitiveType::Unit),
                    span,
                ))],
                span,
            ),
        ]
    }

    /// `statements` when a hook is set.
    pub(super) fn if_hooked(&self, statements: Vec<St>, span: Span) -> St {
        let i64_t = prim(PrimitiveType::I64);
        if_(
            binary(
                BinaryOp::Ne,
                var(intern(library::debug::HOOK_MASK), i64_t.clone(), span),
                int_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            statements,
            None,
            span,
        )
    }

    /// A call of a library function the debug library's hooks and
    /// frames go through, when this function keeps a frame.
    pub(super) fn debug_call(&self, name: &str, args: Vec<Node>, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        vec![expr_stmt(call(name, args, prim(PrimitiveType::Unit), span))]
    }

    /// A loop is about to run its body again: the line its head is on
    /// runs (`zl_dbg_loop`), or for a `while`, the jump back lands on
    /// its head (`back`).
    pub(super) fn debug_loop_back(&self, head: Span, back: bool, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        let line = self.m.line_of(head);
        let name = if back {
            "zl_dbg_loop_head"
        } else {
            "zl_dbg_loop"
        };
        vec![self.if_hooked(
            vec![expr_stmt(call(
                name,
                vec![int_lit(line, span)],
                prim(PrimitiveType::Unit),
                span,
            ))],
            span,
        )]
    }

    /// A loop whose head is on `head` was left; `end_line` is told to
    /// a line hook as the line after it, for a generic `for`.
    pub(super) fn debug_loop_exit(&self, head: Span, end_line: Option<i64>, span: Span) -> Vec<St> {
        if !self.frames {
            return Vec::new();
        }
        let line = self.m.line_of(head);
        let mut hooked = vec![expr_stmt(call(
            "zl_dbg_loop_exit",
            vec![int_lit(line, span)],
            prim(PrimitiveType::Unit),
            span,
        ))];
        if let Some(end) = end_line {
            hooked.push(expr_stmt(call(
                "zl_dbg_line",
                vec![int_lit(end, span)],
                prim(PrimitiveType::Unit),
                span,
            )));
        }
        vec![self.if_hooked(hooked, span)]
    }

    /// `zl_dbg_enter(key)`: this function's frame, pushed.
    pub(super) fn debug_enter(&self, f: FuncId, span: Span) -> St {
        expr_stmt(call(
            "zl_dbg_enter",
            vec![int_lit(self.m.func_key(f), span)],
            prim(PrimitiveType::Unit),
            span,
        ))
    }

    /// What the debug library is told of this function: where it is,
    /// its parameters and upvalues, the lines its statements start on
    /// and its parameters' names. A stripped function has no active
    /// lines, no parameter names and upvalues named `(no name)`.
    pub(super) fn debug_function_records(&mut self, id: FuncId, line: i64, last_line: i64) {
        let mask = (1i64 << library::LINE_BITS) - 1;
        let info = self.scopes().func(id).clone();
        let stripped = self.m.stripped;
        let mut f = vec![
            "F".to_string(),
            id.0.to_string(),
            (line & mask).to_string(),
            (last_line & mask).to_string(),
            info.params.len().to_string(),
            if info.is_vararg { "1" } else { "0" }.to_string(),
        ];
        f.extend(info.upvalues.iter().map(|u| match u {
            _ if stripped => "(no name)".to_string(),
            crate::scope::Upvalue::Var(v) => self.scopes().var(*v).name.clone(),
            crate::scope::Upvalue::Env => "_ENV".to_string(),
        }));
        self.m.debug_record(&f);
        let mut lines = std::mem::take(&mut self.lines);
        if stripped {
            lines.clear();
        } else if id != CHUNK {
            lines.insert(last_line & mask);
        }
        let mut a = vec!["A".to_string(), id.0.to_string()];
        a.extend(lines.iter().map(|l| l.to_string()));
        self.m.debug_record(&a);
        let mut l = vec!["L".to_string(), id.0.to_string(), "0".to_string()];
        if !stripped {
            l.extend(
                info.params
                    .iter()
                    .map(|v| self.scopes().var(*v).name.clone()),
            );
        }
        self.m.debug_record(&l);
    }

    /// The names of the locals in scope, as `debug.getlocal` gives them:
    /// `(temporary)` each in a stripped chunk.
    pub(super) fn live_names(&self) -> Vec<String> {
        self.live
            .iter()
            .map(|l| match l {
                _ if self.m.stripped => "(temporary)".to_string(),
                Live::Var(v) => self.scopes().var(*v).name.clone(),
                Live::ForState(_) => "(for state)".to_string(),
            })
            .collect()
    }

    /// The live locals kept in cells, a bit each by slot (the first 63
    /// slots): a spill holds such a local's cell, not its value, so a
    /// `debug.setlocal` reaches the closures sharing it at once.
    pub(super) fn live_cells(&mut self) -> i64 {
        let mut mask = 0i64;
        for (i, l) in self.live.clone().into_iter().enumerate().take(63) {
            if let Live::Var(v) = l
                && matches!(self.storage_of(v), Storage::Cell(..))
            {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// A call site of this function, numbered for the debug library,
    /// with what the callee is called there (`desc`); the locals in
    /// scope recorded for `debug.getlocal`. None when this function
    /// keeps no frame.
    pub(super) fn debug_site(&mut self, desc: &Desc, span: Span) -> Option<i64> {
        self.debug_site_as(desc, "", span)
    }

    /// [`Self::debug_site`] for a callee a traceback names `global`
    /// whatever the site calls it (a library function), unless that is
    /// empty.
    pub(super) fn debug_site_as(&mut self, desc: &Desc, global: &str, span: Span) -> Option<i64> {
        if !self.frames {
            return None;
        }
        let m = self.m;
        let debug = m.debug.as_ref()?;
        let k = debug.sites.get() + 1;
        debug.sites.set(k);
        let (namewhat, name) = match desc.as_deref().and_then(|d| d.split_once(" '")) {
            Some((what, rest)) => (what.to_string(), rest.trim_end_matches('\'').to_string()),
            None => (String::new(), String::new()),
        };
        let global = if !global.is_empty() {
            global.to_string()
        } else if namewhat == "global" {
            name.clone()
        } else {
            String::new()
        };
        m.debug_record(&["S".to_string(), k.to_string(), namewhat, name, global]);
        if debug.locals {
            let mut record = vec!["L".to_string(), self.func.0.to_string(), k.to_string()];
            record.extend(self.live_names());
            m.debug_record(&record);
            let cells = self.live_cells();
            if cells != 0 {
                m.debug_record(&[
                    "C".to_string(),
                    self.func.0.to_string(),
                    k.to_string(),
                    cells.to_string(),
                ]);
            }
        }
        let tail = if self.tail_span == Some(span) {
            library::debug::TAIL_SITE
        } else {
            0
        };
        Some((m.chunk_index << 32) | k | tail)
    }

    /// What a call site does before its call, the arguments evaluated:
    /// the locals spilled (for `debug.getlocal`), the site stored, and
    /// the line the call starts on, which the arguments may have moved
    /// the frame off. The spill list's variable is returned, for
    /// [`Self::after_call`].
    pub(super) fn before_call(
        &mut self,
        desc: &Desc,
        span: Span,
        pre: &mut Vec<St>,
    ) -> Option<InternedString> {
        let site = self.debug_site(desc, span)?;
        let spill = self.spill(site, span, pre);
        pre.push(self.set_line(span));
        pre.push(assign(
            var(
                intern(library::debug::DBG_SITE),
                prim(PrimitiveType::I64),
                span,
            ),
            int_lit(site, span),
            span,
        ));
        spill
    }

    /// The locals in scope (and a variadic function's extra arguments,
    /// after them) as a list handed to the host for the frame, when
    /// the program reads locals.
    pub(super) fn spill(
        &mut self,
        site: i64,
        span: Span,
        pre: &mut Vec<St>,
    ) -> Option<InternedString> {
        if !self.debug_locals() {
            return None;
        }
        let mut items = Vec::with_capacity(self.live.len() + 1);
        let cells = self.live_cells();
        for (i, l) in self.live.clone().into_iter().enumerate() {
            items.push(match l {
                Live::Var(v) if i < 63 && cells & (1 << i) != 0 => self.capture_value(v, span),
                Live::Var(v) => {
                    let value = self.read_var(v, span);
                    self.boxed(value)
                }
                Live::ForState(i) => self.for_states[i].clone(),
            });
        }
        if let Some(varargs) = self.varargs
            && self.func != CHUNK
        {
            items.push(call(
                "zb_box_tuple",
                vec![var(varargs, self.m.anys(), span)],
                Type::Any,
                span,
            ));
        }
        let list = self.array_of(items, pre, span);
        let name = self.temp();
        pre.push(let_(
            name,
            Type::Any,
            call("zb_list_box_any", vec![list], Type::Any, span),
            span,
        ));
        pre.push(expr_stmt(call(
            "zl_dbg_spill",
            vec![var(name, Type::Any, span), int_lit(site, span)],
            prim(PrimitiveType::Unit),
            span,
        )));
        Some(name)
    }

    /// What a call site does once its call returned: the locals read
    /// back from the list they were spilled to, when a callee may
    /// have set them, and the list released.
    pub(super) fn after_call(&mut self, spill: Option<InternedString>, span: Span) -> Vec<St> {
        let Some(spill) = spill else {
            return Vec::new();
        };
        let i64_t = prim(PrimitiveType::I64);
        let unspill = call(
            "zl_dbg_unspill",
            vec![var(spill, Type::Any, span)],
            i64_t.clone(),
            span,
        );
        if !self.m.debug.as_ref().is_some_and(|d| d.setlocal) {
            return vec![expr_stmt(unspill)];
        }
        // Each local a callee set is read back; the others keep what
        // the call left in them. Straight-line, as the call may sit in
        // an expression. A local in a cell was set through it.
        let set = self.temp();
        let mut out = vec![let_(set, i64_t.clone(), unspill, span)];
        let cells = self.live_cells();
        for (i, l) in self.live.clone().into_iter().enumerate().take(63) {
            let Live::Var(v) = l else {
                continue;
            };
            if cells & (1 << i) != 0 {
                continue;
            }
            let current = self.read_var(v, span);
            let current = self.boxed(current);
            let value = Val {
                node: call(
                    "zl_dbg_pick",
                    vec![
                        var(spill, Type::Any, span),
                        var(set, i64_t.clone(), span),
                        int_lit(i as i64, span),
                        current,
                    ],
                    Type::Any,
                    span,
                ),
                ty: Ty::Any,
            };
            out.push(self.write_var(v, value, span));
        }
        out
    }
}
