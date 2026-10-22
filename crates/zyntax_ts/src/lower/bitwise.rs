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

    /// `left % right`. Positive integers below 2^63 take an inline integer remainder, which has
    /// the dividend's sign and never produces a negative zero for a positive dividend; every
    /// other operand (negative, zero, fractional, NaN, infinite, huge) keeps the exact `fmod`.
    /// The float-to-integer conversions run only after the range test, since an out-of-range
    /// conversion traps in compiled code.
    pub(super) fn remainder(&mut self, left: Expr, right: Expr, at: oxc_span::Span) -> Expr {
        let (x, y) = (self.fresh("dividend"), self.fresh("divisor"));
        let (xi, yi) = (self.fresh("whole_dividend"), self.fresh("whole_divisor"));
        let read = |name| var(name, number(), at);
        let whole = |name| var(name, integer(), at);
        let let_ = |name, ty: Type, value: Expr| {
            stmt(
                TypedStatement::Let(TypedLet {
                    name,
                    ty,
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value)),
                    span: span(at),
                }),
                at,
            )
        };
        let and = |a, b| binary(BinaryOp::And, a, b, boolean(), at);
        let compare = |op, name, bound: f64| binary(op, read(name), num(bound, at), boolean(), at);
        let in_range = |name| {
            and(
                compare(BinaryOp::Gt, name, 0.0),
                compare(BinaryOp::Lt, name, LIMIT),
            )
        };
        let integral = |value: InternedString, whole_value: InternedString| {
            binary(
                BinaryOp::Eq,
                cast(whole(whole_value), number(), at),
                read(value),
                boolean(),
                at,
            )
        };
        let exact = || binary(BinaryOp::Rem, read(x), read(y), number(), at);
        let inner = node(
            TypedExpression::Block(TypedBlock {
                statements: vec![
                    let_(xi, integer(), cast(read(x), integer(), at)),
                    let_(yi, integer(), cast(read(y), integer(), at)),
                    expression(node(
                        TypedExpression::If(TypedIfExpr {
                            condition: Box::new(and(integral(x, xi), integral(y, yi))),
                            then_branch: Box::new(cast(
                                op(BinaryOp::Rem, whole(xi), whole(yi), at),
                                number(),
                                at,
                            )),
                            else_branch: Box::new(exact()),
                        }),
                        number(),
                        at,
                    )),
                ],
                span: span(at),
            }),
            number(),
            at,
        );
        let statements = vec![
            let_(x, number(), left),
            let_(y, number(), right),
            expression(node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(and(in_range(x), in_range(y))),
                    then_branch: Box::new(inner),
                    else_branch: Box::new(exact()),
                }),
                number(),
                at,
            )),
        ];
        node(
            TypedExpression::Block(TypedBlock {
                statements,
                span: span(at),
            }),
            number(),
            at,
        )
    }

    pub(super) fn bitwise_not(&mut self, value: Expr, at: oxc_span::Span) -> Result<Expr> {
        self.expect(&value.ty, &number(), at)?;
        let a = self.to_int32(value, at);
        Ok(cast(op(BinaryOp::BitXor, a, int(-1, at), at), number(), at))
    }
}
