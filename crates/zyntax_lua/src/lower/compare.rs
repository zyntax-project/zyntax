//! Compare for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// The text a string or number contributes to `..`.
    pub(super) fn text_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => v.node,
            Ty::Int => call(
                "zb_str_of_int",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            Ty::Float => call(
                "zl_float_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            Ty::Number => {
                let str_t = prim(PrimitiveType::String);
                let (pre, n) = self.number_parts(v);
                let text = if_value(
                    n.is_int(span),
                    call("zb_str_of_int", vec![n.int], str_t.clone(), span),
                    call("zl_float_str", vec![n.float], str_t.clone(), span),
                    str_t,
                    span,
                );
                block_value(pre, text, span)
            }
            _ => {
                let b = self.boxed(v);
                self.guard(Val {
                    node: call("zl_concat_text", vec![b], prim(PrimitiveType::String), span),
                    ty: Ty::Str,
                })
                .node
            }
        }
    }

    pub(super) fn equal(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        if or_nil_comparable(a.ty, b.ty) {
            return self.or_nil_eq(a, b, span);
        }
        match (a.ty, b.ty) {
            (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Bool, Ty::Bool) => {
                binary(BinaryOp::Eq, a.node, b.node, bool_t, span)
            }
            (Ty::Int, Ty::Float) => mixed_numeric_compare(a.node, b.node, Compare::Eq, true, span),
            (Ty::Float, Ty::Int) => mixed_numeric_compare(b.node, a.node, Compare::Eq, false, span),
            (x, y) if x.is_number() && y.is_number() => {
                self.number_compare(a, b, Compare::Eq, span)
            }
            (x, y) if (x.is_tagged() || y.is_tagged()) && x.is_scalar() && y.is_scalar() => {
                self.scalar_eq(a, b, span)
            }
            // Tables are equal when they are the same table, `__eq`
            // aside; a shaped table may be nil.
            (Ty::Shape(_), Ty::Nil) => binary(
                BinaryOp::Eq,
                a.node,
                block_value(
                    vec![expr_stmt(b.node)],
                    null(self.ir(Ty::Table), span),
                    span,
                ),
                bool_t,
                span,
            ),
            (Ty::Nil, Ty::Shape(_)) => binary(
                BinaryOp::Eq,
                block_value(
                    vec![expr_stmt(a.node)],
                    null(self.ir(Ty::Table), span),
                    span,
                ),
                b.node,
                bool_t,
                span,
            ),
            (Ty::Func(_), Ty::Nil) | (Ty::Nil, Ty::Func(_)) => {
                binary(BinaryOp::Eq, a.node, b.node, bool_t, span)
            }
            (Ty::Str, Ty::Str) => call("zb_str_eq", vec![a.node, b.node], bool_t, span),
            (Ty::Nil, Ty::Nil) => block_value(
                vec![expr_stmt(a.node), expr_stmt(b.node)],
                bool_lit(true, span),
                span,
            ),
            // Two different known types are never equal; the operands
            // still run.
            (x, y)
                if x != Ty::Any
                    && y != Ty::Any
                    && x != y
                    && !matches!(x, Ty::Table | Ty::Shape(_) | Ty::Func(_))
                    && !matches!(y, Ty::Table | Ty::Shape(_) | Ty::Func(_)) =>
            {
                block_value(
                    vec![expr_stmt(a.node), expr_stmt(b.node)],
                    bool_lit(false, span),
                    span,
                )
            }
            _ => {
                let x = self.boxed(a);
                let y = self.boxed(b);
                call("zl_eq", vec![x, y], bool_t, span)
            }
        }
    }

    pub(super) fn less(&mut self, a: Val, b: Val, or_equal: bool, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let op = if or_equal { BinaryOp::Le } else { BinaryOp::Lt };
        if a.ty.is_number() && b.ty.is_number() {
            if a.ty == Ty::Number || b.ty == Ty::Number {
                let how = if or_equal { Compare::Le } else { Compare::Lt };
                return self.number_compare(a, b, how, span);
            }
            if a.ty == b.ty {
                return binary(op, a.node, b.node, bool_t, span);
            }
            let how = if or_equal { Compare::Le } else { Compare::Lt };
            return if a.ty == Ty::Int {
                mixed_numeric_compare(a.node, b.node, how, true, span)
            } else {
                mixed_numeric_compare(b.node, a.node, how, false, span)
            };
        }
        if a.ty == Ty::Str && b.ty == Ty::Str {
            let cmp = call(
                "zb_str_cmp",
                vec![a.node, b.node],
                prim(PrimitiveType::I32),
                span,
            );
            return binary(op, cmp, int32_lit(0, span), bool_t, span);
        }
        let x = self.boxed(a);
        let y = self.boxed(b);
        call(
            if or_equal { "zl_le" } else { "zl_lt" },
            vec![x, y],
            bool_t,
            span,
        )
    }

    /// `a and b` is `a` when that is false, else `b`; `a or b` the
    /// other way round. The right side runs only when needed.
    pub(super) fn logical(
        &mut self,
        is_and: bool,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let bool_t = prim(PrimitiveType::Bool);
        if a.ty == Ty::Bool && b.ty == Ty::Bool {
            let op = if is_and { BinaryOp::And } else { BinaryOp::Or };
            return Ok(Val {
                node: binary(op, a.node, b.node, bool_t, span),
                ty: Ty::Bool,
            });
        }
        // The result type is the join; `a` is read twice, so it is held.
        let ty = types::logical_ty(is_and, a.ty, b.ty).settled();
        let mut pre = Vec::new();
        let a = self.hold(a, &mut pre);
        let test = self.truthy(a.clone());
        let a_node = self.coerce(a, ty);
        let b_node = self.coerce(b, ty);
        let (then, els) = if is_and {
            (b_node, a_node)
        } else {
            (a_node, b_node)
        };
        let value = if_value(test, then, els, self.ir(ty), span);
        Ok(Val {
            node: block_value(pre, value, span),
            ty,
        })
    }
}
