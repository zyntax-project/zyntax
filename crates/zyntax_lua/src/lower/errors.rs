//! Errors for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The placeholder this function leaves with once an error is
    /// pending: whoever called it checks next.
    pub(super) fn placeholder_return(&mut self, span: Span) -> St {
        if !self.tbc.is_empty() {
            // The error has left the function when its `<close>`
            // variables are closed: its frame goes first.
            let mut statements = self.debug_call("zl_dbg_leave", Vec::new(), span);
            self.closes_from(1, span, &mut statements);
            statements.push(self.placeholder_return_plain(span, false));
            return stmt(TypedStatement::Block(TypedBlock { statements, span }), span);
        }
        self.placeholder_return_plain(span, true)
    }

    /// Leaving with the placeholder, the depth uncounted and, when
    /// `pop`, the frame popped.
    pub(super) fn placeholder_return_plain(&mut self, span: Span, pop: bool) -> St {
        let leave = self.placeholder_value(span);
        if self.counts_depth || (self.frames && pop) {
            let mut statements = Vec::new();
            if self.counts_depth {
                statements.push(depth_step(-1, span));
            }
            if pop {
                statements.extend(self.debug_call("zl_dbg_leave", Vec::new(), span));
            }
            statements.push(leave);
            return stmt(TypedStatement::Block(TypedBlock { statements, span }), span);
        }
        leave
    }

    pub(super) fn placeholder_value(&mut self, span: Span) -> St {
        let value = match self.returns.clone() {
            Returns::Fixed(types) if types.is_empty() => None,
            Returns::Fixed(types) if types.len() == 1 => {
                Some(self.zero_of(types[0].settled(), span))
            }
            Returns::Fixed(_) => Some(node(
                TypedExpression::Array(Vec::new()),
                self.m.anys(),
                span,
            )),
            Returns::Dynamic | Returns::Unknown => Some(nil(span)),
        };
        ret(value, span)
    }

    pub(super) fn zero_of(&mut self, ty: Ty, span: Span) -> Node {
        match ty {
            Ty::Int => int_lit(0, span),
            Ty::Float => float_lit(0.0, span),
            Ty::Number => number_of_int(int_lit(0, span), span),
            Ty::Scalar | Ty::IntOrNil | Ty::FloatOrNil => scalar_of_nil(span),
            Ty::Bool => bool_lit(false, span),
            Ty::Str => str_lit("", span),
            Ty::Shape(_) => null(self.ir(Ty::Table), span),
            Ty::Table => call("zl_table_new", vec![], self.ir(Ty::Table), span),
            _ => nil(span),
        }
    }

    pub(super) fn pending(span: Span) -> Node {
        var(intern(library::PENDING), Type::Any, span)
    }

    /// `if an error is pending, leave`; then the line is this one
    /// again, since a callee sets its own (unless nothing reads it
    /// before it is stored again: see [`LineRestores`]). A type error's
    /// note of which operand it is about is cleared on the way: it was
    /// a deeper site's.
    pub(super) fn pending_check(&mut self, span: Span) -> St {
        self.pending_check_described(span, &Described::NONE)
    }

    /// The same, the error described first: a type error about an
    /// operand of this site gets that operand's description appended,
    /// `(local 'x')` and the like, the way the reference names the
    /// variable.
    pub(super) fn pending_check_described(&mut self, span: Span, descs: &Described) -> St {
        let cond = binary(
            BinaryOp::Ne,
            Self::pending(span),
            nil(span),
            prim(PrimitiveType::Bool),
            span,
        );
        let leave = self.leave_described(span, descs);
        let restore = self.set_line(span);
        if_(cond, leave, Some(vec![restore]), span)
    }

    /// Leaving with the pending error, described as above.
    pub(super) fn leave_described(&mut self, span: Span, descs: &Described) -> Vec<St> {
        self.raised = true;
        self.line_needed = true;
        let settle = if descs.operands.iter().all(Option::is_none) {
            assign(
                var(intern(library::VARINFO), prim(PrimitiveType::I64), span),
                int_lit(0, span),
                span,
            )
        } else {
            let text = |d: &Desc| {
                let d = d.as_deref().map(|d| format!(" ({d})")).unwrap_or_default();
                str_lit(&d, span)
            };
            expr_stmt(call(
                "zl_annotate",
                vec![
                    text(&descs.operands[0]),
                    text(&descs.operands[1]),
                    bool_lit(descs.call, span),
                ],
                Type::Primitive(PrimitiveType::Unit),
                span,
            ))
        };
        let leave = self.placeholder_return(span);
        vec![settle, leave]
    }

    /// `zl_depth += 1; if zl_depth > limit { raise; leave }`: one more
    /// frame, refused past what the reference allows.
    pub(super) fn stack_check(&mut self, span: Span) -> St {
        let i64_t = prim(PrimitiveType::I64);
        let depth = var(intern(library::DEPTH), i64_t, span);
        let over = binary(
            BinaryOp::Gt,
            depth,
            int_lit(library::MAX_DEPTH, span),
            prim(PrimitiveType::Bool),
            span,
        );
        let leave = self.placeholder_return(span);
        let statements = vec![
            depth_step(1, span),
            if_(
                over,
                vec![
                    expr_stmt(call(
                        "zl_stack_overflow",
                        vec![],
                        prim(PrimitiveType::Unit),
                        span,
                    )),
                    leave,
                ],
                None,
                span,
            ),
        ];
        stmt(TypedStatement::Block(TypedBlock { statements, span }), span)
    }

    /// `zl_line = <the line of span>`.
    pub(super) fn set_line(&mut self, span: Span) -> St {
        let line = self.m.line_of(span);
        assign(
            var(intern(library::LINE), prim(PrimitiveType::I64), span),
            int_lit(line, span),
            span,
        )
    }

    /// Whether a call node names a library function that can raise.
    pub(super) fn call_can_raise(&self, node: &Node) -> bool {
        let TypedExpression::Call(c) = &node.node else {
            return false;
        };
        let TypedExpression::Variable(name) = &c.callee.node else {
            return false;
        };
        name.resolve_global()
            .is_some_and(|n| self.m.fallible.contains(n.as_str()))
    }

    /// A value from a call that may have raised: held, then checked
    /// before anything uses it. A call that cannot raise is left alone.
    pub(super) fn guarded_described(&mut self, v: Val, descs: &Described) -> Val {
        if !self.call_can_raise(&v.node) {
            return v;
        }
        self.guard_described(v, descs)
    }

    pub(super) fn guard(&mut self, v: Val) -> Val {
        self.guard_described(v, &Described::NONE)
    }

    pub(super) fn guard_described(&mut self, v: Val, descs: &Described) -> Val {
        let span = v.node.span;
        let mut pre = Vec::new();
        let held = if v.node.ty == prim(PrimitiveType::Unit) {
            pre.push(expr_stmt(v.node));
            Val {
                node: nil(span),
                ty: Ty::Nil,
            }
        } else {
            self.hold(v, &mut pre)
        };
        pre.push(self.pending_check_described(span, descs));
        Val {
            node: block_value(pre, held.node, span),
            ty: held.ty,
        }
    }

    /// A statement calling something that may have raised, checked.
    pub(super) fn guarded_stmt(&mut self, node: Node, descs: &Described) -> St {
        if !self.call_can_raise(&node) {
            return expr_stmt(node);
        }
        let span = node.span;
        let mut statements = vec![expr_stmt(node)];
        statements.push(self.pending_check_described(span, descs));
        stmt(TypedStatement::Block(TypedBlock { statements, span }), span)
    }
}
