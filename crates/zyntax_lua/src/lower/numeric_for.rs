//! Numeric for for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A loop body. A label ending it is where a `goto` continues to,
    /// and lands ahead of whatever the loop does after the body.
    pub(super) fn loop_body(&mut self, block: &Block, end: Span) -> Result<Vec<St>> {
        self.loop_depths.push((self.depth + 1, end));
        let body = self.block(block);
        self.loop_depths.pop();
        body
    }

    /// A `for` loop's body: the loop's hidden locals, `states` their
    /// values boxed (read only by `debug.getlocal`), then its
    /// variables, in scope. `end` is the loop's `end`.
    pub(super) fn for_body(
        &mut self,
        block: &Block,
        states: Vec<Node>,
        vars: &[VarId],
        end: Span,
    ) -> Result<Vec<St>> {
        let live = self.live.len();
        let held = self.for_states.len();
        for state in states {
            self.live.push(Live::ForState(self.for_states.len()));
            self.for_states.push(state);
        }
        self.live.extend(vars.iter().map(|v| Live::Var(*v)));
        let body = self.loop_body(block, end);
        self.live.truncate(live);
        self.for_states.truncate(held);
        body
    }

    /// Whether the program reads locals through the debug library, so
    /// call sites spill them.
    pub(super) fn debug_locals(&self) -> bool {
        self.m.debug.as_ref().is_some_and(|d| d.locals)
    }

    pub(super) fn numeric_for(
        &mut self,
        f: &ast::NumericFor,
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let v = self.scopes().declared(f.index_variable());
        let start = self.expr(f.start())?;
        let limit = self.expr(f.end())?;
        let step = f.step().map(|s| self.expr(s)).transpose()?;
        let loop_ty = types::numeric_for_ty(start.ty, step.as_ref().map_or(Ty::Int, |s| s.ty));
        if !matches!(loop_ty, Ty::Int | Ty::Float) {
            return self.numeric_for_dynamic(f, v, (start, limit, step), span, out);
        }
        let is_float = loop_ty == Ty::Float;
        let num_ty = if is_float { Ty::Float } else { Ty::Int };
        let ir = self.ir(num_ty);
        let bool_t = prim(PrimitiveType::Bool);
        // A limit or start of another type is converted as Lua would:
        // a float limit of an integer loop is the last integer the
        // loop may reach, which depends on the step's direction.
        let start_node = match start.ty {
            t if t == num_ty => start.node,
            Ty::Int | Ty::Float => self.coerce(start, num_ty),
            _ => {
                let b = self.boxed(start);
                let node = if is_float {
                    call(
                        "zl_for_float",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                } else {
                    call(
                        "zl_for_int",
                        vec![b, str_lit("initial value", span)],
                        ir.clone(),
                        span,
                    )
                };
                self.guard(Val { node, ty: num_ty }).node
            }
        };
        let limit_literal = match &limit.node.node {
            TypedExpression::Literal(TypedLiteral::Integer(n)) if limit.ty == Ty::Int => {
                Some(*n as i64)
            }
            _ => None,
        };
        // A limit converted with the step in hand is evaluated before
        // the step all the same.
        enum Limit {
            Ready(Node),
            Float(Node),
            Dynamic(Node),
        }
        let limit = match limit.ty {
            t if t == num_ty => Limit::Ready(limit.node),
            Ty::Int => Limit::Ready(self.coerce(limit, num_ty)),
            Ty::Float => Limit::Float(limit.node),
            _ if is_float => {
                let b = self.boxed(limit);
                let node = call(
                    "zl_for_float",
                    vec![b, str_lit("limit", span)],
                    ir.clone(),
                    span,
                );
                Limit::Ready(self.guard(Val { node, ty: num_ty }).node)
            }
            _ => Limit::Dynamic(self.boxed(limit)),
        };
        // The step's sign, when it is a literal, picks the test.
        let (step_node, step_literal): (Node, Option<f64>) = match step {
            None => (
                if is_float {
                    float_lit(1.0, span)
                } else {
                    int_lit(1, span)
                },
                Some(1.0),
            ),
            Some(s) => {
                let literal = match &s.node.node {
                    TypedExpression::Literal(TypedLiteral::Integer(n)) => Some(*n as f64),
                    TypedExpression::Literal(TypedLiteral::Float(x)) => Some(*x),
                    _ => None,
                };
                let n = match s.ty {
                    t if t == num_ty => s.node,
                    Ty::Int | Ty::Float => self.coerce(s, num_ty),
                    _ => {
                        let b = self.boxed(s);
                        let node = if is_float {
                            call(
                                "zl_for_float",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        } else {
                            call(
                                "zl_for_int",
                                vec![b, str_lit("step", span)],
                                ir.clone(),
                                span,
                            )
                        };
                        self.guard(Val { node, ty: num_ty }).node
                    }
                };
                (n, literal)
            }
        };
        let counter = self.temp();
        let lim = self.temp();
        let st = self.temp();
        out.push(let_(counter, ir.clone(), start_node, span));
        let converted = match limit {
            Limit::Ready(node) => {
                out.push(let_(lim, ir.clone(), node, span));
                out.push(let_(st, ir.clone(), step_node, span));
                false
            }
            Limit::Float(node) => {
                let raw = self.temp();
                let f64_t = prim(PrimitiveType::F64);
                out.push(let_(raw, f64_t.clone(), node, span));
                out.push(let_(st, ir.clone(), step_node, span));
                let node = call(
                    "zl_for_limit_f",
                    vec![var(raw, f64_t, span), var(st, ir.clone(), span)],
                    ir.clone(),
                    span,
                );
                out.push(let_(lim, ir.clone(), node, span));
                true
            }
            Limit::Dynamic(b) => {
                let raw = self.temp();
                out.push(let_(raw, Type::Any, b, span));
                out.push(let_(st, ir.clone(), step_node, span));
                let node = call(
                    "zl_for_limit",
                    vec![var(raw, Type::Any, span), var(st, ir.clone(), span)],
                    ir.clone(),
                    span,
                );
                let node = self.guard(Val { node, ty: num_ty }).node;
                out.push(let_(lim, ir.clone(), node, span));
                true
            }
        };
        if converted {
            // A limit past the integers against the step: the counter
            // is put past the (clamped) limit, so the loop does not run.
            let past = if_value(
                binary(
                    BinaryOp::Gt,
                    var(st, ir.clone(), span),
                    int_lit(0, span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Add,
                    var(lim, ir.clone(), span),
                    int_lit(1, span),
                    ir.clone(),
                    span,
                ),
                binary(
                    BinaryOp::Sub,
                    var(lim, ir.clone(), span),
                    int_lit(1, span),
                    ir.clone(),
                    span,
                ),
                ir.clone(),
                span,
            );
            out.push(if_(
                var(intern(library::FOR_SKIP), bool_t.clone(), span),
                vec![assign(var(counter, ir.clone(), span), past, span)],
                None,
                span,
            ));
        }
        let zero = if is_float {
            float_lit(0.0, span)
        } else {
            int_lit(0, span)
        };
        let counter_v = || var(counter, ir.clone(), span);
        let lim_v = || var(lim, ir.clone(), span);
        let st_v = || var(st, ir.clone(), span);
        // A zero step is an error, a literal one at once.
        if step_literal.is_none_or(|x| x == 0.0) {
            let raise = expr_stmt(call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit("'for' step is zero", span)],
                prim(PrimitiveType::Unit),
                span,
            ));
            let leave = self.pending_check(span);
            if step_literal.is_none() {
                out.push(if_(
                    binary(BinaryOp::Eq, st_v(), zero.clone(), bool_t.clone(), span),
                    vec![raise, leave],
                    None,
                    span,
                ));
            } else {
                out.push(raise);
                out.push(leave);
            }
        }
        let cond = match step_literal {
            Some(x) if x > 0.0 => binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
            Some(_) => binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
            None => binary(
                BinaryOp::Or,
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Gt, st_v(), zero.clone(), bool_t.clone(), span),
                    binary(BinaryOp::Le, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Lt, st_v(), zero, bool_t.clone(), span),
                    binary(BinaryOp::Ge, counter_v(), lim_v(), bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            ),
        };
        // An integer counter must not run past the limit into a wrap:
        // the loop leaves at the limit itself, unless the limit is a
        // literal the step cannot carry past the range.
        let bounded = !is_float
            && !matches!(
                (limit_literal, step_literal),
                (Some(l), Some(s)) if (s as i64).checked_add(l).is_some() && l.checked_add(s as i64).is_some()
                    && (s as i64) != i64::MIN
            );
        // The body's variable is a copy of the counter each iteration.
        let mut body = vec![self.declare_var(
            v,
            Val {
                node: counter_v(),
                ty: num_ty,
            },
            span,
        )];
        let states = self.numeric_states((counter_v(), lim_v(), st_v()), is_float, span);
        let mut inner = self.for_body(f.block(), states, &[v], span_of(f.end_token()))?;
        inner.extend(self.debug_loop_back(span_of(f.for_token()), false, span));
        let mut after = Vec::new();
        if bounded {
            // Whether the step fits between the counter and the limit:
            // the distance and the step's size as unsigned values, so
            // neither wraps.
            let u64_t = prim(PrimitiveType::U64);
            let distance = |from: Node, to: Node| {
                cast(
                    binary(BinaryOp::Sub, to, from, ir.clone(), span),
                    u64_t.clone(),
                    span,
                )
            };
            let up = || distance(counter_v(), lim_v());
            let down = || distance(lim_v(), counter_v());
            let size = |negated: bool| {
                let s = if negated {
                    binary(BinaryOp::Sub, int_lit(0, span), st_v(), ir.clone(), span)
                } else {
                    st_v()
                };
                cast(s, u64_t.clone(), span)
            };
            let short =
                |gap: Node, step: Node| binary(BinaryOp::Lt, gap, step, bool_t.clone(), span);
            let done = match step_literal {
                Some(x) if x == 1.0 || x == -1.0 => {
                    binary(BinaryOp::Eq, counter_v(), lim_v(), bool_t.clone(), span)
                }
                Some(x) if x > 0.0 => short(up(), size(false)),
                Some(_) => short(down(), size(true)),
                None => if_value(
                    binary(BinaryOp::Gt, st_v(), int_lit(0, span), bool_t.clone(), span),
                    short(up(), size(false)),
                    short(down(), size(true)),
                    bool_t.clone(),
                    span,
                ),
            };
            after.push(if_(
                done,
                vec![stmt(TypedStatement::Break(None), span)],
                None,
                span,
            ));
        }
        after.push(assign(
            counter_v(),
            binary(BinaryOp::Add, counter_v(), st_v(), ir.clone(), span),
            span,
        ));
        body.extend(inner);
        body.extend(after);
        out.push(while_(cond, body, span));
        out.extend(self.debug_loop_exit(span_of(f.for_token()), None, span));
        Ok(())
    }

    /// A numeric loop's hidden locals as Lua keeps them, for
    /// `debug.getlocal`: the index, then for an integer loop the
    /// iterations left after this one and for a float loop the limit,
    /// then the step. `parts` are the counter, the limit and the step.
    pub(super) fn numeric_states(
        &mut self,
        parts: (Node, Node, Node),
        is_float: bool,
        span: Span,
    ) -> Vec<Node> {
        if !self.debug_locals() {
            return vec![nil(span); 3];
        }
        let (counter, lim, st) = parts;
        let num_ty = if is_float { Ty::Float } else { Ty::Int };
        let rest = if is_float {
            lim
        } else {
            Self::iterations_left(counter.clone(), lim, st.clone(), span)
        };
        [counter, rest, st]
            .into_iter()
            .map(|node| self.boxed(Val { node, ty: num_ty }))
            .collect()
    }

    /// How many more times an integer loop runs, counted as Lua counts
    /// it: the distance to the limit over the step's size, unsigned.
    pub(super) fn iterations_left(counter: Node, lim: Node, st: Node, span: Span) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let u64_t = prim(PrimitiveType::U64);
        let bool_t = prim(PrimitiveType::Bool);
        let up = binary(BinaryOp::Gt, st.clone(), int_lit(0, span), bool_t, span);
        let distance = if_value(
            up.clone(),
            binary(
                BinaryOp::Sub,
                lim.clone(),
                counter.clone(),
                i64_t.clone(),
                span,
            ),
            binary(BinaryOp::Sub, counter, lim, i64_t.clone(), span),
            i64_t.clone(),
            span,
        );
        // `-(step + 1) + 1`, so the smallest integer step does not wrap.
        let down = binary(
            BinaryOp::Add,
            cast(
                binary(
                    BinaryOp::Sub,
                    int_lit(-1, span),
                    st.clone(),
                    i64_t.clone(),
                    span,
                ),
                u64_t.clone(),
                span,
            ),
            cast(int_lit(1, span), u64_t.clone(), span),
            u64_t.clone(),
            span,
        );
        let size = if_value(up, cast(st, u64_t.clone(), span), down, u64_t.clone(), span);
        cast(
            binary(
                BinaryOp::Div,
                cast(distance, u64_t.clone(), span),
                size,
                u64_t,
                span,
            ),
            i64_t,
            span,
        )
    }
}
