//! Sort for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The sort for `table.sort` over tables of shape `k` by the
    /// function `f`: the library's quicksort over the array part, the
    /// comparator called directly with the elements as their type.
    /// `lua$sort$<k>$<f>(arr, lo, hi, line, rec)`, `line` the sort's
    /// own for the error it raises, `rec` the comparator's record.
    pub(super) fn sort_helper(&mut self, k: ShapeId, f: FuncId, span: Span) -> String {
        let key = (k, f);
        if let Some(helper) = self.m.sorts.borrow().get(&key) {
            return helper.clone();
        }
        let helper = format!("lua${}sort${}${}", self.m.tag, k.0, f.0);
        let mut lowerer = Lowerer::new(self.m, CHUNK);
        lowerer.returns = Returns::Fixed(Vec::new());
        let anys = self.m.anys();
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let unit = prim(PrimitiveType::Unit);
        let arr = || var(intern("arr"), anys.clone(), span);
        let lo = || var(intern("lo"), i64_t.clone(), span);
        let hi = || var(intern("hi"), i64_t.clone(), span);
        let line = || var(intern("line"), i64_t.clone(), span);
        let rec = || var(intern("rec"), Type::Any, span);
        let i = intern("i");
        let j = intern("j");
        let pivot = intern("pivot");
        let iv = || var(i, i64_t.clone(), span);
        let jv = || var(j, i64_t.clone(), span);
        let at = |x: Node| index(arr(), x, Type::Any, span);
        let record = Val {
            node: rec(),
            ty: Ty::Any,
        };
        // `less(a, b)`: the comparator's first result, as a truth.
        let less = |lowerer: &mut Lowerer<'m, 'a>, a: Node, b: Node| -> Node {
            let multi = lowerer.direct_call_vals(
                f,
                Some(record.clone()),
                Vec::new(),
                vec![
                    Val {
                        node: a,
                        ty: Ty::Any,
                    },
                    Val {
                        node: b,
                        ty: Ty::Any,
                    },
                ],
                None,
                &None,
                span,
            );
            let first = match multi {
                Multi::Fixed(vals) => vals
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| lowerer.nil_val(span)),
                Multi::Dynamic(node) => Val {
                    node: call("zl_first", vec![node], Type::Any, span),
                    ty: Ty::Any,
                },
                Multi::None(node) => Val {
                    node: block_value(vec![expr_stmt(node)], nil(span), span),
                    ty: Ty::Nil,
                },
            };
            lowerer.truthy(first)
        };
        let step =
            |x: Node, by: i64| binary(BinaryOp::Add, x, int_lit(by, span), i64_t.clone(), span);
        let cmp = |op: BinaryOp, a: Node, b: Node| binary(op, a, b, bool_t.clone(), span);
        let less_i = less(&mut lowerer, at(iv()), var(pivot, Type::Any, span));
        let less_j = less(&mut lowerer, var(pivot, Type::Any, span), at(jv()));
        let tmp = intern("tmp");
        let body = vec![
            if_(
                cmp(BinaryOp::Ge, lo(), hi()),
                vec![ret(None, span)],
                None,
                span,
            ),
            let_(
                pivot,
                Type::Any,
                at(binary(
                    BinaryOp::Div,
                    binary(BinaryOp::Add, lo(), hi(), i64_t.clone(), span),
                    int_lit(2, span),
                    i64_t.clone(),
                    span,
                )),
                span,
            ),
            let_(i, i64_t.clone(), lo(), span),
            let_(j, i64_t.clone(), hi(), span),
            while_(
                cmp(BinaryOp::Le, iv(), jv()),
                vec![
                    // An order that never settles walks off the range:
                    // the comparator is not one.
                    while_(
                        binary(
                            BinaryOp::And,
                            cmp(BinaryOp::Le, iv(), hi()),
                            less_i,
                            bool_t.clone(),
                            span,
                        ),
                        vec![assign(iv(), step(iv(), 1), span)],
                        span,
                    ),
                    while_(
                        binary(
                            BinaryOp::And,
                            cmp(BinaryOp::Ge, jv(), lo()),
                            less_j,
                            bool_t.clone(),
                            span,
                        ),
                        vec![assign(jv(), step(jv(), -1), span)],
                        span,
                    ),
                    if_(
                        binary(
                            BinaryOp::Or,
                            cmp(BinaryOp::Gt, iv(), hi()),
                            cmp(BinaryOp::Lt, jv(), lo()),
                            bool_t.clone(),
                            span,
                        ),
                        vec![
                            assign(
                                var(intern(library::LINE), i64_t.clone(), span),
                                line(),
                                span,
                            ),
                            expr_stmt(call(
                                "zb_fatal",
                                vec![
                                    str_lit("error", span),
                                    str_lit("invalid order function for sorting", span),
                                ],
                                unit.clone(),
                                span,
                            )),
                            ret(None, span),
                        ],
                        None,
                        span,
                    ),
                    if_(
                        cmp(BinaryOp::Le, iv(), jv()),
                        vec![
                            let_(tmp, Type::Any, at(iv()), span),
                            assign(at(iv()), at(jv()), span),
                            assign(at(jv()), var(tmp, Type::Any, span), span),
                            assign(iv(), step(iv(), 1), span),
                            assign(jv(), step(jv(), -1), span),
                        ],
                        None,
                        span,
                    ),
                ],
                span,
            ),
            if_(
                binary(
                    BinaryOp::Ne,
                    Lowerer::pending(span),
                    nil(span),
                    bool_t.clone(),
                    span,
                ),
                vec![ret(None, span)],
                None,
                span,
            ),
            expr_stmt(call(
                &helper,
                vec![arr(), lo(), jv(), line(), rec()],
                unit.clone(),
                span,
            )),
            expr_stmt(call(
                &helper,
                vec![arr(), iv(), hi(), line(), rec()],
                unit,
                span,
            )),
            ret(None, span),
        ];
        let params = vec![
            parameter(intern("arr"), anys, span),
            parameter(intern("lo"), i64_t.clone(), span),
            parameter(intern("hi"), i64_t.clone(), span),
            parameter(intern("line"), i64_t, span),
            parameter(intern("rec"), Type::Any, span),
        ];
        self.m.functions.borrow_mut().push(typed_function(
            &helper,
            params,
            prim(PrimitiveType::Unit),
            body,
            span,
        ));
        self.m.sorts.borrow_mut().insert(key, helper.clone());
        helper
    }
}
