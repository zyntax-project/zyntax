//! Pairs for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// `for k, v in pairs(t)`: every position holding a value, walked
    /// in place. A table whose metatable has `__pairs` is walked as
    /// that says instead: `f, s, c` from the handler, then `f(s, c)`
    /// until its first value is nil. The body is lowered once, so one
    /// loop serves both, choosing its step by the handler's presence.
    pub(super) fn pairs_loop(
        &mut self,
        names: &[VarId],
        t: Val,
        f: &ast::GenericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let block = f.block();
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let table_t = self.ir(Ty::Table);
        let anys_t = self.m.anys();
        let tname = self.temp();
        let t = if t.ty == Ty::Table {
            t.node
        } else {
            let b = self.boxed(t);
            let node = call(
                "zl_as_table",
                vec![b, str_lit("for iterator", span)],
                table_t.clone(),
                span,
            );
            self.guard(Val {
                node,
                ty: Ty::Table,
            })
            .node
        };
        out.push(let_(tname, table_t.clone(), t, span));
        let tv = || var(tname, table_t.clone(), span);
        // The handler, or nil.
        let handler = self.temp();
        out.push(let_(
            handler,
            Type::Any,
            call(
                "zl_meta",
                vec![tv(), str_lit("__pairs", span)],
                Type::Any,
                span,
            ),
            span,
        ));
        let hv = || var(handler, Type::Any, span);
        let walked = || binary(BinaryOp::Eq, hv(), nil(span), bool_t.clone(), span);
        // The walk: the position, and the array part's length at the
        // last step, since a body that removes elements shortens it,
        // moving the positions after.
        let pos = self.temp();
        let seen = self.temp();
        out.push(let_(pos, i64_t.clone(), int_lit(-1, span), span));
        out.push(let_(seen, i64_t.clone(), int_lit(0, span), span));
        // The protocol: `f`, `s` and the control `c`.
        let (fname, sname, cname) = (self.temp(), self.temp(), self.temp());
        for name in [fname, sname, cname] {
            out.push(let_(name, Type::Any, nil(span), span));
        }
        let array_len = || call("zl_len", vec![tv()], i64_t.clone(), span);
        let value_at = |vals: InternedString, i: i64| {
            call(
                "zl_value_at",
                vec![var(vals, anys_t.clone(), span), int_lit(i, span)],
                Type::Any,
                span,
            )
        };
        let triple = self.temp();
        let from_handler = {
            let b = self.box_table(tv());
            let called = call("zl_call_1", vec![hv(), b], Type::Any, span);
            let called = self
                .guard(Val {
                    node: called,
                    ty: Ty::Any,
                })
                .node;
            vec![
                let_(
                    triple,
                    anys_t.clone(),
                    call("zl_values", vec![called], anys_t.clone(), span),
                    span,
                ),
                assign(var(fname, Type::Any, span), value_at(triple, 1), span),
                assign(var(sname, Type::Any, span), value_at(triple, 2), span),
                assign(var(cname, Type::Any, span), value_at(triple, 3), span),
            ]
        };
        let mut walk = vec![
            assign(
                var(pos, i64_t.clone(), span),
                call(
                    "zl_next_pos",
                    vec![tv(), int_lit(0, span)],
                    i64_t.clone(),
                    span,
                ),
                span,
            ),
            assign(var(seen, i64_t.clone(), span), array_len(), span),
        ];
        // The walk keeps what `pairs(t)` would have given, `next` and
        // `t`, where `debug.getlocal` reads the loop's state.
        if self.debug_locals() {
            let next = crate::library::stdlib::BUILTINS
                .iter()
                .find(|b| b.lib.is_empty() && b.name == "next")
                .expect("the base library has next");
            let next = self.builtin_value(next, span).node;
            let t = self.box_table(tv());
            walk.push(assign(var(fname, Type::Any, span), next, span));
            walk.push(assign(var(sname, Type::Any, span), t, span));
        }
        out.push(if_(walked(), walk, Some(from_handler), span));
        // Each iteration's key and value.
        let (key, value) = (self.temp(), self.temp());
        let mut body = vec![
            let_(key, Type::Any, nil(span), span),
            let_(value, Type::Any, nil(span), span),
        ];
        let leave = || stmt(TypedStatement::Break(None), span);
        let from_position = vec![
            if_(
                binary(
                    BinaryOp::Lt,
                    var(pos, i64_t.clone(), span),
                    int_lit(0, span),
                    bool_t.clone(),
                    span,
                ),
                vec![leave()],
                None,
                span,
            ),
            // The value first: making the key may collect, which clears
            // a weak value nothing holds yet.
            assign(
                var(value, Type::Any, span),
                call(
                    "zl_pos_value",
                    vec![tv(), var(pos, i64_t.clone(), span)],
                    Type::Any,
                    span,
                ),
                span,
            ),
            // A collection since the step found this position cleared
            // a weak value there: the walk goes on from the next live
            // position, found and read with nothing between that can
            // collect.
            if_(
                binary(
                    BinaryOp::Eq,
                    var(value, Type::Any, span),
                    nil(span),
                    bool_t.clone(),
                    span,
                ),
                vec![
                    assign(
                        var(pos, i64_t.clone(), span),
                        call(
                            "zl_next_pos",
                            vec![
                                tv(),
                                binary(
                                    BinaryOp::Add,
                                    var(pos, i64_t.clone(), span),
                                    int_lit(1, span),
                                    i64_t.clone(),
                                    span,
                                ),
                            ],
                            i64_t.clone(),
                            span,
                        ),
                        span,
                    ),
                    if_(
                        binary(
                            BinaryOp::Lt,
                            var(pos, i64_t.clone(), span),
                            int_lit(0, span),
                            bool_t.clone(),
                            span,
                        ),
                        vec![leave()],
                        None,
                        span,
                    ),
                    assign(
                        var(value, Type::Any, span),
                        call(
                            "zl_pos_value",
                            vec![tv(), var(pos, i64_t.clone(), span)],
                            Type::Any,
                            span,
                        ),
                        span,
                    ),
                ],
                None,
                span,
            ),
            assign(
                var(key, Type::Any, span),
                call(
                    "zl_pos_key",
                    vec![tv(), var(pos, i64_t.clone(), span)],
                    Type::Any,
                    span,
                ),
                span,
            ),
        ];
        let vals = self.temp();
        let from_protocol = {
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
            vec![
                let_(
                    vals,
                    anys_t.clone(),
                    call("zl_values", vec![step], anys_t.clone(), span),
                    span,
                ),
                assign(var(key, Type::Any, span), value_at(vals, 1), span),
                if_(
                    binary(
                        BinaryOp::Eq,
                        var(key, Type::Any, span),
                        nil(span),
                        bool_t.clone(),
                        span,
                    ),
                    vec![leave()],
                    None,
                    span,
                ),
                assign(var(value, Type::Any, span), value_at(vals, 2), span),
                assign(var(cname, Type::Any, span), var(key, Type::Any, span), span),
            ]
        };
        body.push(if_(walked(), from_position, Some(from_protocol), span));
        for (i, v) in names.iter().enumerate() {
            let node = match i {
                0 => var(key, Type::Any, span),
                1 => var(value, Type::Any, span),
                _ => nil(span),
            };
            let val = Val {
                node,
                ty: if i < 2 { Ty::Any } else { Ty::Nil },
            };
            body.push(self.declare_var(*v, val, span));
        }
        let states = vec![
            var(fname, Type::Any, span),
            var(sname, Type::Any, span),
            var(key, Type::Any, span),
            nil(span),
        ];
        let mut inner = self.for_body(block, states, names, span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let step = assign(
            var(pos, i64_t.clone(), span),
            call(
                "zl_next_pos_from",
                vec![
                    tv(),
                    binary(
                        BinaryOp::Add,
                        var(pos, i64_t.clone(), span),
                        int_lit(1, span),
                        i64_t.clone(),
                        span,
                    ),
                    var(seen, i64_t.clone(), span),
                ],
                i64_t.clone(),
                span,
            ),
            span,
        );
        let remember = assign(var(seen, i64_t.clone(), span), array_len(), span);
        let advance = if_(walked(), vec![step, remember], None, span);
        let mut whole = body;
        whole.extend(inner);
        whole.push(advance);
        out.push(while_(bool_lit(true, span), whole, span));
        let end = self.m.line_of(span_of(f.end_token()));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), Some(end), span));
        Ok(())
    }

    /// A `while true` loop whose body ends in an increment; a label
    /// ending the body lands ahead of it.
    pub(super) fn push_loop_with_increment(
        &mut self,
        head: Vec<St>,
        inner: Vec<St>,
        increment: St,
        span: Span,
        out: &mut Vec<St>,
    ) {
        let mut body = head;
        body.extend(inner);
        body.push(increment);
        out.push(while_(bool_lit(true, span), body, span));
    }
}
