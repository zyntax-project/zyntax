//! Binary for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn binary_op(
        &mut self,
        op: &BinOp,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<Val> {
        // `and` and `or` evaluate their right side conditionally.
        match op {
            BinOp::And(_) | BinOp::Or(_) => {
                return self.logical(matches!(op, BinOp::And(_)), lhs, rhs, span);
            }
            _ => {}
        }
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let descs = Described::operands(self.describe(lhs), self.describe(rhs));
        let v = self.binary_vals(op, a, b, &descs, span)?;
        Ok(self.guarded_described(v, &descs))
    }

    pub(super) fn binary_vals(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        descs: &Described,
        span: Span,
    ) -> Result<Val> {
        // A shaped table operand whose handler the types know.
        if let Some(event) = types::arith_event(op)
            && let Some(sides) = self.typer().metamethod_sides(event, a.ty, b.ty)
            && !sides.is_empty()
        {
            let code = match op {
                BinOp::Plus(_) => OP_ADD,
                BinOp::Minus(_) => OP_SUB,
                BinOp::Star(_) => OP_MUL,
                BinOp::Slash(_) => OP_DIV,
                BinOp::Percent(_) => OP_MOD,
                BinOp::Caret(_) => OP_POW,
                _ => OP_IDIV,
            };
            let mut pre = Vec::new();
            let a = self.hold(a, &mut pre);
            let b = self.hold(b, &mut pre);
            let x = self.boxed(a.clone());
            let y = self.boxed(b.clone());
            let generic = self.guard_described(
                Val {
                    node: call("zl_arith", vec![int_lit(code, span), x, y], Type::Any, span),
                    ty: Ty::Any,
                },
                descs,
            );
            let ty = self
                .typer()
                .metamethod_result(event, a.ty, b.ty)
                .unwrap_or(Ty::Any)
                .settled();
            let value = self.metamethod_dispatch(event, sides, vec![a, b], generic, ty, span);
            return Ok(Val {
                node: block_value(pre, value, span),
                ty,
            });
        }
        let ints = a.ty == Ty::Int && b.ty == Ty::Int;
        let numbers = a.ty.is_number() && b.ty.is_number();
        // A literal divisor that is not 0 or -1 makes `//` and `%`
        // plain arithmetic: nothing to raise, nothing to check after.
        let literal_divisor = match &b.node.node {
            TypedExpression::Literal(TypedLiteral::Integer(n)) if b.ty == Ty::Int => {
                Some(*n as i64)
            }
            _ => None,
        };
        let plain_divisor = literal_divisor.is_some_and(|n| n != 0 && n != -1);
        // A positive power of two divides a float exactly, so its
        // modulo is three operations instead of fmod; an integer's
        // floor division and modulo by it are an arithmetic shift and a
        // mask, for either sign.
        let pow2_divisor = literal_divisor.filter(|n| *n > 0 && n & (n - 1) == 0);
        let divisor = Divisor {
            plain: plain_divisor,
            pow2: pow2_divisor,
        };
        // A float operand makes the result a float whatever the other
        // number is, so only comparisons and the operators with an
        // integer result still need the other's kind.
        let float_result_op = matches!(
            op,
            BinOp::Plus(_)
                | BinOp::Minus(_)
                | BinOp::Star(_)
                | BinOp::DoubleSlash(_)
                | BinOp::Percent(_)
        ) && (a.ty == Ty::Float || b.ty == Ty::Float);
        // A number whose kind is not known decides it at run time.
        if numbers
            && !float_result_op
            && (a.ty == Ty::Number || b.ty == Ty::Number)
            && takes_number_path(op)
        {
            return Ok(self.number_binary(op, a, b, divisor, span));
        }
        // A scalar that may not be a number: arithmetic and ordering
        // raise for nil and booleans as the dynamic value does, then
        // take the number path.
        let arithmetic_or_order = matches!(
            op,
            BinOp::Plus(_)
                | BinOp::Minus(_)
                | BinOp::Star(_)
                | BinOp::Slash(_)
                | BinOp::Caret(_)
                | BinOp::DoubleSlash(_)
                | BinOp::Percent(_)
                | BinOp::LessThan(_)
                | BinOp::LessThanEqual(_)
                | BinOp::GreaterThan(_)
                | BinOp::GreaterThanEqual(_)
        );
        if arithmetic_or_order && or_nil_operands(a.ty, b.ty) {
            return self.or_nil_binary(op, a, b, descs, span);
        }
        if arithmetic_or_order
            && (a.ty.is_tagged() || b.ty.is_tagged())
            && a.ty.is_scalar()
            && b.ty.is_scalar()
        {
            return Ok(self.scalar_binary(op, a, b, divisor, descs, span));
        }
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let as_float = |this: &mut Self, v: Val| this.coerce(v, Ty::Float);
        let int_result = |n: Node| Val {
            node: n,
            ty: Ty::Int,
        };
        let float_result = |n: Node| Val {
            node: n,
            ty: Ty::Float,
        };
        let bool_result = |n: Node| Val {
            node: n,
            ty: Ty::Bool,
        };
        let arith = |this: &mut Self, code: i64, a: Val, b: Val| {
            let a = this.boxed(a);
            let b = this.boxed(b);
            Val {
                node: call("zl_arith", vec![int_lit(code, span), a, b], Type::Any, span),
                ty: Ty::Any,
            }
        };
        Ok(match op {
            BinOp::Plus(_) | BinOp::Minus(_) | BinOp::Star(_) => {
                let bop = match op {
                    BinOp::Plus(_) => BinaryOp::Add,
                    BinOp::Minus(_) => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                let code = match op {
                    BinOp::Plus(_) => OP_ADD,
                    BinOp::Minus(_) => OP_SUB,
                    _ => OP_MUL,
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(bop, x, y, f64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::Slash(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(binary(BinaryOp::Div, x, y, f64_t, span))
                } else {
                    arith(self, OP_DIV, a, b)
                }
            }
            BinOp::Caret(_) => {
                if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_pow", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_POW, a, b)
                }
            }
            BinOp::DoubleSlash(_) => {
                if let (true, Some(n)) = (ints, pow2_divisor) {
                    int_result(pow2_floordiv(a.node, n, span))
                } else if ints && plain_divisor {
                    int_result(call("zb_floordiv_i64", vec![a.node, b.node], i64_t, span))
                } else if ints {
                    int_result(call("zl_idiv_i64", vec![a.node, b.node], i64_t, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_idiv_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_IDIV, a, b)
                }
            }
            BinOp::Percent(_) => {
                if let (true, Some(n)) = (ints, pow2_divisor) {
                    int_result(pow2_mod(a.node, n, span))
                } else if ints && plain_divisor {
                    int_result(call("zb_mod_i64", vec![a.node, b.node], i64_t, span))
                } else if ints {
                    int_result(call("zl_mod_i64", vec![a.node, b.node], i64_t, span))
                } else if let (true, Some(n)) = (numbers, pow2_divisor) {
                    let x = as_float(self, a);
                    float_result(self.float_mod_pow2(x, n, span))
                } else if numbers {
                    let (x, y) = (as_float(self, a), as_float(self, b));
                    float_result(call("zl_mod_f64", vec![x, y], f64_t, span))
                } else {
                    arith(self, OP_MOD, a, b)
                }
            }
            BinOp::Ampersand(_) | BinOp::Pipe(_) | BinOp::Tilde(_) => {
                let (bop, code) = match op {
                    BinOp::Ampersand(_) => (BinaryOp::BitAnd, OP_BAND),
                    BinOp::Pipe(_) => (BinaryOp::BitOr, OP_BOR),
                    _ => (BinaryOp::BitXor, OP_BXOR),
                };
                if ints {
                    int_result(binary(bop, a.node, b.node, i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::DoubleLessThan(_) | BinOp::DoubleGreaterThan(_) => {
                let (f, code) = match op {
                    BinOp::DoubleLessThan(_) => ("zl_shl_i64", OP_SHL),
                    _ => ("zl_shr_i64", OP_SHR),
                };
                if ints {
                    int_result(call(f, vec![a.node, b.node], i64_t, span))
                } else {
                    arith(self, code, a, b)
                }
            }
            BinOp::TwoDots(_) => {
                let text = |t: Ty| matches!(t, Ty::Str | Ty::Int | Ty::Float);
                if text(a.ty) && text(b.ty) {
                    let x = self.text_of(a);
                    let y = self.text_of(b);
                    Val {
                        node: binary(BinaryOp::Add, x, y, prim(PrimitiveType::String), span),
                        ty: Ty::Str,
                    }
                } else {
                    let x = self.boxed(a);
                    let y = self.boxed(b);
                    Val {
                        node: call("zl_concat", vec![x, y], Type::Any, span),
                        ty: Ty::Any,
                    }
                }
            }
            BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                let negate = matches!(op, BinOp::TildeEqual(_));
                let eq = self.equal(a, b, span);
                bool_result(if negate {
                    unary(UnaryOp::Not, eq, bool_t, span)
                } else {
                    eq
                })
            }
            BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_) => {
                // `a > b` is `b < a`.
                let (a, b, or_equal) = match op {
                    BinOp::LessThan(_) => (a, b, false),
                    BinOp::LessThanEqual(_) => (a, b, true),
                    BinOp::GreaterThan(_) => (b, a, false),
                    _ => (b, a, true),
                };
                bool_result(self.less(a, b, or_equal, span))
            }
            _ => return unsupported("this binary operator", span),
        })
    }
}
