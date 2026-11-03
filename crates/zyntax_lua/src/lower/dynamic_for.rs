//! Dynamic for for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A numeric `for` whose start or step is not known to be an
    /// integer or a float: an integer loop when both are integers when
    /// it starts, else a float loop, the variable a number either way.
    /// `values` are the start, the limit and the step, lowered in that
    /// order.
    pub(super) fn numeric_for_dynamic(
        &mut self,
        f: &ast::NumericFor,
        v: VarId,
        values: (Val, Val, Option<Val>),
        span: Span,
        out: &mut Vec<St>,
    ) -> Result<()> {
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let (start, limit, step) = values;
        let step = step.unwrap_or(Val {
            node: int_lit(1, span),
            ty: Ty::Int,
        });
        let mut boxed = Vec::new();
        for value in [start, limit, step] {
            let b = self.boxed(value);
            let name = self.temp();
            out.push(let_(name, Type::Any, b, span));
            boxed.push(var(name, Type::Any, span));
        }
        let [start, limit, step] = <[Node; 3]>::try_from(boxed).expect("three values");
        let ints = self.temp();
        out.push(let_(
            ints,
            bool_t.clone(),
            call(
                "zl_for_ints",
                vec![start.clone(), step.clone()],
                bool_t.clone(),
                span,
            ),
            span,
        ));
        let ints_v = || var(ints, bool_t.clone(), span);
        let [ci, li, si, cf, lf, sf] = [(); 6].map(|_| self.temp());
        for (name, ty, zero) in [
            (ci, &i64_t, int_lit(0, span)),
            (li, &i64_t, int_lit(0, span)),
            (si, &i64_t, int_lit(0, span)),
            (cf, &f64_t, float_lit(0.0, span)),
            (lf, &f64_t, float_lit(0.0, span)),
            (sf, &f64_t, float_lit(0.0, span)),
        ] {
            out.push(let_(name, ty.clone(), zero, span));
        }
        let int_v = |name| var(name, i64_t.clone(), span);
        let float_v = |name| var(name, f64_t.clone(), span);
        let converted = |this: &mut Self, helper: &str, value: &Node, what: &str, ty: Ty| {
            let node = call(
                helper,
                vec![value.clone(), str_lit(what, span)],
                this.ir(ty),
                span,
            );
            this.guard(Val { node, ty }).node
        };
        let step_zero = |this: &mut Self, is_zero: Node| {
            let raise = expr_stmt(call(
                "zb_fatal",
                vec![str_lit("error", span), str_lit("'for' step is zero", span)],
                prim(PrimitiveType::Unit),
                span,
            ));
            let leave = this.pending_check(span);
            if_(is_zero, vec![raise, leave], None, span)
        };
        // Lua's order: the integer loop checks the step and then the
        // limit; the float loop converts the limit, the step and the
        // start, then checks the step.
        let mut int_prep = vec![assign(
            int_v(si),
            converted(self, "zl_for_int", &step, "step", Ty::Int),
            span,
        )];
        int_prep.push(step_zero(
            self,
            binary(
                BinaryOp::Eq,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
        ));
        int_prep.push(assign(
            int_v(ci),
            converted(self, "zl_for_int", &start, "initial value", Ty::Int),
            span,
        ));
        let lim = call(
            "zl_for_limit",
            vec![limit.clone(), int_v(si)],
            i64_t.clone(),
            span,
        );
        let lim = self.guard(Val {
            node: lim,
            ty: Ty::Int,
        });
        int_prep.push(assign(int_v(li), lim.node, span));
        // A limit past the integers against the step: the counter is
        // put past the (clamped) limit, so the loop does not run.
        let past = if_value(
            binary(
                BinaryOp::Gt,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Add,
                int_v(li),
                int_lit(1, span),
                i64_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Sub,
                int_v(li),
                int_lit(1, span),
                i64_t.clone(),
                span,
            ),
            i64_t.clone(),
            span,
        );
        int_prep.push(if_(
            var(intern(library::FOR_SKIP), bool_t.clone(), span),
            vec![assign(int_v(ci), past, span)],
            None,
            span,
        ));
        let mut float_prep = vec![
            assign(
                float_v(lf),
                converted(self, "zl_for_float", &limit, "limit", Ty::Float),
                span,
            ),
            assign(
                float_v(sf),
                converted(self, "zl_for_float", &step, "step", Ty::Float),
                span,
            ),
            assign(
                float_v(cf),
                converted(self, "zl_for_float", &start, "initial value", Ty::Float),
                span,
            ),
        ];
        float_prep.push(step_zero(
            self,
            binary(
                BinaryOp::Eq,
                float_v(sf),
                float_lit(0.0, span),
                bool_t.clone(),
                span,
            ),
        ));
        out.push(if_(ints_v(), int_prep, Some(float_prep), span));
        // Whether the counter is within the limit, the step's sign
        // deciding which side.
        let within = |counter: Node, lim: Node, st: Node, zero: Node| {
            binary(
                BinaryOp::Or,
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Gt, st.clone(), zero.clone(), bool_t.clone(), span),
                    binary(
                        BinaryOp::Le,
                        counter.clone(),
                        lim.clone(),
                        bool_t.clone(),
                        span,
                    ),
                    bool_t.clone(),
                    span,
                ),
                binary(
                    BinaryOp::And,
                    binary(BinaryOp::Lt, st, zero, bool_t.clone(), span),
                    binary(BinaryOp::Ge, counter, lim, bool_t.clone(), span),
                    bool_t.clone(),
                    span,
                ),
                bool_t.clone(),
                span,
            )
        };
        let cond = if_value(
            ints_v(),
            within(int_v(ci), int_v(li), int_v(si), int_lit(0, span)),
            within(float_v(cf), float_v(lf), float_v(sf), float_lit(0.0, span)),
            bool_t.clone(),
            span,
        );
        let value = if_value(
            ints_v(),
            number_of_int(int_v(ci), span),
            number_of_float(float_v(cf), span),
            number_type(),
            span,
        );
        let mut body = vec![self.declare_var(
            v,
            Val {
                node: value,
                ty: Ty::Number,
            },
            span,
        )];
        let states = if self.debug_locals() {
            let parts = [
                (int_v(ci), float_v(cf)),
                (
                    Self::iterations_left(int_v(ci), int_v(li), int_v(si), span),
                    float_v(lf),
                ),
                (int_v(si), float_v(sf)),
            ];
            parts
                .into_iter()
                .map(|(int, float)| {
                    let node = if_value(
                        ints_v(),
                        number_of_int(int, span),
                        number_of_float(float, span),
                        number_type(),
                        span,
                    );
                    self.boxed(Val {
                        node,
                        ty: Ty::Number,
                    })
                })
                .collect()
        } else {
            vec![nil(span); 3]
        };
        body.extend(self.for_body(f.block(), states, &[v], span_of(f.end_token()))?);
        // The integer counter leaves when the step would carry it past
        // the limit, so it never wraps: the distance and the step's
        // size compared as unsigned values.
        let u64_t = prim(PrimitiveType::U64);
        let unsigned = |n: Node| cast(n, u64_t.clone(), span);
        let short = |gap: Node, size: Node| {
            binary(
                BinaryOp::Lt,
                unsigned(gap),
                unsigned(size),
                bool_t.clone(),
                span,
            )
        };
        let done = if_value(
            binary(
                BinaryOp::Gt,
                int_v(si),
                int_lit(0, span),
                bool_t.clone(),
                span,
            ),
            short(
                binary(BinaryOp::Sub, int_v(li), int_v(ci), i64_t.clone(), span),
                int_v(si),
            ),
            short(
                binary(BinaryOp::Sub, int_v(ci), int_v(li), i64_t.clone(), span),
                binary(
                    BinaryOp::Sub,
                    int_lit(0, span),
                    int_v(si),
                    i64_t.clone(),
                    span,
                ),
            ),
            bool_t.clone(),
            span,
        );
        let int_step = vec![
            if_(
                done,
                vec![stmt(TypedStatement::Break(None), span)],
                None,
                span,
            ),
            assign(
                int_v(ci),
                binary(BinaryOp::Add, int_v(ci), int_v(si), i64_t.clone(), span),
                span,
            ),
        ];
        let float_step = vec![assign(
            float_v(cf),
            binary(BinaryOp::Add, float_v(cf), float_v(sf), f64_t.clone(), span),
            span,
        )];
        body.push(if_(ints_v(), int_step, Some(float_step), span));
        out.push(while_(cond, body, span));
        Ok(())
    }
}
