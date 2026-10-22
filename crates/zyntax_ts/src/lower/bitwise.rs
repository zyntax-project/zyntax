//! ECMAScript bitwise operators. Operands convert with ToInt32 and all arithmetic runs on
//! sign-extended 64-bit integers, so no narrower integer types are involved.

use super::*;

const LIMIT: f64 = 9.0e18;

fn integer() -> Type {
    Type::Primitive(PrimitiveType::I64)
}

fn int(value: i128, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(value)),
        integer(),
        at,
    )
}

fn op(operator: BinaryOp, left: Expr, right: Expr, at: oxc_span::Span) -> Expr {
    binary(operator, left, right, integer(), at)
}

fn cast(value: Expr, ty: Type, at: oxc_span::Span) -> Expr {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ty.clone(),
        }),
        ty,
        at,
    )
}

/// Keeps the low 32 bits, sign-extended.
fn sign_extend(value: Expr, at: oxc_span::Span) -> Expr {
    let high = op(BinaryOp::Shl, value, int(32, at), at);
    op(BinaryOp::Shr, high, int(32, at), at)
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// ToInt32 of a number, as a sign-extended `i64`.
    fn to_int32(&mut self, value: Expr, at: oxc_span::Span) -> Expr {
        let name = self.fresh("number");
        let x = || var(name, number(), at);
        let in_range = binary(
            BinaryOp::And,
            binary(BinaryOp::Ge, x(), num(-LIMIT, at), boolean(), at),
            binary(BinaryOp::Le, x(), num(LIMIT, at), boolean(), at),
            boolean(),
            at,
        );
        // NaN, infinities and magnitudes beyond i64 take the exact path.
        let exact = self.call("$TS$to_int32_slow", vec![x()], integer(), at);
        let wide = node(
            TypedExpression::If(TypedIfExpr {
                condition: Box::new(in_range),
                then_branch: Box::new(cast(x(), integer(), at)),
                else_branch: Box::new(exact),
            }),
            integer(),
            at,
        );
        let statements = vec![
            stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: number(),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            ),
            expression(sign_extend(wide, at)),
        ];
        node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            integer(),
            at,
        )
    }

    pub(super) fn bitwise(
        &mut self,
        operator: &str,
        left: Expr,
        right: Expr,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        self.expect(&left.ty, &number(), at)?;
        self.expect(&right.ty, &number(), at)?;
        let a = self.to_int32(left, at);
        let b = self.to_int32(right, at);
        // Shift counts use the low five bits.
        let count = |b: Expr| op(BinaryOp::BitAnd, b, int(31, at), at);
        let result = match operator {
            "|" => op(BinaryOp::BitOr, a, b, at),
            "&" => op(BinaryOp::BitAnd, a, b, at),
            "^" => op(BinaryOp::BitXor, a, b, at),
            "<<" => sign_extend(op(BinaryOp::Shl, a, count(b), at), at),
            ">>" => op(BinaryOp::Shr, a, count(b), at),
            ">>>" => {
                let unsigned = op(BinaryOp::BitAnd, a, int(0xFFFF_FFFF, at), at);
                op(BinaryOp::Shr, unsigned, count(b), at)
            }
            _ => return self.unsupported("this bitwise operator", at),
        };
        Ok(cast(result, number(), at))
    }

    pub(super) fn bitwise_not(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        self.expect(&value.ty, &number(), at)?;
        let a = self.to_int32(value, at);
        Ok(cast(op(BinaryOp::BitXor, a, int(-1, at), at), number(), at))
    }
}
