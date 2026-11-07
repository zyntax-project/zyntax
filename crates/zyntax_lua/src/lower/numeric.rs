//! Numeric for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// Arithmetic or ordering on scalars of which at least one may be
    /// nil or a boolean: the dynamic operation runs when either is,
    /// which raises; then both are numbers and the number path runs.
    pub(super) fn scalar_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        divisor: Divisor,
        descs: &Described,
        span: Span,
    ) -> Val {
        let mut pre = Vec::new();
        let a = Val {
            node: self.coerce(a, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let b = Val {
            node: self.coerce(b, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let ha = self.hold(a, &mut pre);
        let hb = self.hold(b, &mut pre);
        let x = NumberParts::of(&ha.node, span);
        let y = NumberParts::of(&hb.node, span);
        let numeric = x.both_numeric(&y, span);
        let raise = self.scalar_raise(op, &ha, &hb, span);
        pre.push(if_(
            numeric,
            Vec::new(),
            Some(vec![
                expr_stmt(raise),
                self.pending_check_described(span, descs),
            ]),
            span,
        ));
        let na = Val {
            node: ha.node,
            ty: Ty::Number,
        };
        let nb = Val {
            node: hb.node,
            ty: Ty::Number,
        };
        let v = self.number_binary(op, na, nb, divisor, span);
        Val {
            node: block_value(pre, v.node, span),
            ty: v.ty,
        }
    }

    /// The dynamic operation on held scalars `a` and `b` that are not
    /// both numbers: it raises, as on the dynamic values.
    pub(super) fn scalar_raise(&mut self, op: &BinOp, a: &Val, b: &Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let boxed_a = self.coerce(a.clone(), Ty::Any);
        let boxed_b = self.coerce(b.clone(), Ty::Any);
        match op {
            BinOp::LessThan(_) => call("zl_lt", vec![boxed_a, boxed_b], bool_t, span),
            BinOp::LessThanEqual(_) => call("zl_le", vec![boxed_a, boxed_b], bool_t, span),
            BinOp::GreaterThan(_) => call("zl_lt", vec![boxed_b, boxed_a], bool_t, span),
            BinOp::GreaterThanEqual(_) => call("zl_le", vec![boxed_b, boxed_a], bool_t, span),
            _ => {
                let code = match op {
                    BinOp::Plus(_) => OP_ADD,
                    BinOp::Minus(_) => OP_SUB,
                    BinOp::Star(_) => OP_MUL,
                    BinOp::Slash(_) => OP_DIV,
                    BinOp::Caret(_) => OP_POW,
                    BinOp::DoubleSlash(_) => OP_IDIV,
                    _ => OP_MOD,
                };
                call(
                    "zl_arith",
                    vec![int_lit(code, span), boxed_a, boxed_b],
                    Type::Any,
                    span,
                )
            }
        }
    }

    /// Arithmetic or ordering where each operand is a number of known
    /// kind or such a number or nil: a nil operand takes the dynamic
    /// operation, which raises; else the numbers' own operation runs.
    pub(super) fn or_nil_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        descs: &Described,
        span: Span,
    ) -> Result<Val> {
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let ha = self.hold(a, &mut pre);
        let hb = self.hold(b, &mut pre);
        // The tags of the sides that may be nil, and the tag each has
        // when it is its number.
        let mut tags = Vec::new();
        let mut number = |h: &Val| -> Val {
            let n = NumberParts::of(&h.node, span);
            match h.ty {
                Ty::IntOrNil => {
                    tags.push((n.tag, TAG_INT));
                    Val {
                        node: n.int,
                        ty: Ty::Int,
                    }
                }
                Ty::FloatOrNil => {
                    tags.push((n.tag, TAG_FLOAT));
                    Val {
                        node: n.float,
                        ty: Ty::Float,
                    }
                }
                _ => h.clone(),
            }
        };
        let x = number(&ha);
        let y = number(&hb);
        // Both numbers without a short-circuit: tags are at most
        // `TAG_FLOAT`, so a product of two says which pair they are.
        let (tag, expected) = tags
            .into_iter()
            .reduce(|(p, e), (q, f)| (binary(BinaryOp::Mul, p, q, i64_t.clone(), span), e * f))
            .expect("an operand that may be nil");
        let numeric = binary(BinaryOp::Eq, tag, int_lit(expected, span), bool_t, span);
        let raise = self.scalar_raise(op, &ha, &hb, span);
        pre.push(if_(
            numeric,
            Vec::new(),
            Some(vec![
                expr_stmt(raise),
                self.pending_check_described(span, descs),
            ]),
            span,
        ));
        let v = self.binary_vals(op, x, y, descs, span)?;
        Ok(Val {
            node: block_value(pre, v.node, span),
            ty: v.ty,
        })
    }

    /// `a == b` where one side is a number of known kind or nil and the
    /// other that number, nil, or the same: the tags and the number's
    /// part decide, since a scalar's unused parts are zero.
    pub(super) fn or_nil_eq(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let mut pre = Vec::new();
        let a = self.hold(a, &mut pre);
        let b = self.hold(b, &mut pre);
        let (a, b) = if a.ty.is_tagged() { (a, b) } else { (b, a) };
        let (tag, part_ty) = match a.ty {
            Ty::IntOrNil => (TAG_INT, Ty::Int),
            _ => (TAG_FLOAT, Ty::Float),
        };
        let part = |n: NumberParts| if part_ty == Ty::Int { n.int } else { n.float };
        let x = NumberParts::of(&a.node, span);
        let eq = match b.ty {
            Ty::Nil => x.has_tag(TAG_NIL, span),
            t if t == part_ty => {
                let is = x.has_tag(tag, span);
                let same = binary(BinaryOp::Eq, part(x), b.node, bool_t.clone(), span);
                binary(BinaryOp::And, is, same, bool_t, span)
            }
            _ => {
                let y = NumberParts::of(&b.node, span);
                let tags = binary(
                    BinaryOp::Eq,
                    x.tag.clone(),
                    y.tag.clone(),
                    bool_t.clone(),
                    span,
                );
                let same = binary(BinaryOp::Eq, part(x), part(y), bool_t.clone(), span);
                binary(BinaryOp::And, tags, same, bool_t, span)
            }
        };
        block_value(pre, eq, span)
    }

    /// Equality of scalars: numbers compare as numbers, anything else
    /// by tag and payload.
    pub(super) fn scalar_eq(&mut self, a: Val, b: Val, span: Span) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let a = Val {
            node: self.coerce(a, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let b = Val {
            node: self.coerce(b, Ty::Scalar),
            ty: Ty::Scalar,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let numeric = x.both_numeric(&y, span);
        let as_numbers = self.number_compare_parts(&x, &y, Compare::Eq, span);
        let same = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Eq,
                x.tag.clone(),
                y.tag.clone(),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Eq,
                x.int.clone(),
                y.int.clone(),
                bool_t.clone(),
                span,
            ),
            bool_t.clone(),
            span,
        );
        block_value(pre, if_value(numeric, as_numbers, same, bool_t, span), span)
    }

    /// Arithmetic where at least one operand's kind is decided at run
    /// time: both integers takes the integer path, anything else the
    /// float path, and the result says which it took. Bitwise
    /// operators and concatenation are not handled here.
    pub(super) fn number_binary(
        &mut self,
        op: &BinOp,
        a: Val,
        b: Val,
        divisor: Divisor,
        span: Span,
    ) -> Val {
        let plain_divisor = divisor.plain;
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let both_int = |x: &NumberParts, y: &NumberParts| x.both_int(y, span);
        let a = Val {
            node: self.coerce(a, Ty::Number),
            ty: Ty::Number,
        };
        let b = Val {
            node: self.coerce(b, Ty::Number),
            ty: Ty::Number,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let number = |pre: Vec<St>, int: Node, float: Node, is_int: Node| Val {
            node: block_value(
                pre,
                number_value(tag_of_is_int(is_int, span), int, float, span),
                span,
            ),
            ty: Ty::Number,
        };
        match op {
            BinOp::Plus(_) | BinOp::Minus(_) | BinOp::Star(_) => {
                let bop = match op {
                    BinOp::Plus(_) => BinaryOp::Add,
                    BinOp::Minus(_) => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                // Both paths are pure, so both run and the tag picks;
                // an integer result carries its value as the float.
                let mut pre = pre;
                let int = binary(bop, x.int.clone(), y.int.clone(), i64_t, span);
                let int = self
                    .hold(
                        Val {
                            node: int,
                            ty: Ty::Int,
                        },
                        &mut pre,
                    )
                    .node;
                let is_int = both_int(&x, &y);
                let float = float_word(
                    is_int.clone(),
                    int.clone(),
                    binary(bop, x.as_float(span), y.as_float(span), f64_t, span),
                    span,
                );
                number(pre, int, float, is_int)
            }
            BinOp::Slash(_) | BinOp::Caret(_) => {
                let (l, r) = (x.as_float(span), y.as_float(span));
                let float = if matches!(op, BinOp::Slash(_)) {
                    binary(BinaryOp::Div, l, r, f64_t, span)
                } else {
                    call("zl_pow", vec![l, r], f64_t, span)
                };
                Val {
                    node: block_value(pre, float, span),
                    ty: Ty::Float,
                }
            }
            BinOp::DoubleSlash(_) | BinOp::Percent(_) => {
                // The integer path raises on a zero divisor, so only
                // the path taken runs.
                let (fi, ff) = match (op, plain_divisor) {
                    (BinOp::DoubleSlash(_), true) => ("zb_floordiv_i64", "zl_idiv_f64"),
                    (BinOp::DoubleSlash(_), false) => ("zl_idiv_i64", "zl_idiv_f64"),
                    (_, true) => ("zb_mod_i64", "zl_mod_f64"),
                    (_, false) => ("zl_mod_i64", "zl_mod_f64"),
                };
                let is_int = both_int(&x, &y);
                let int_part = match (op, divisor.pow2) {
                    (BinOp::Percent(_), Some(n)) => pow2_mod(x.int.clone(), n, span),
                    (_, Some(n)) => pow2_floordiv(x.int.clone(), n, span),
                    _ => call(fi, vec![x.int.clone(), y.int.clone()], i64_t.clone(), span),
                };
                let int = if_value(is_int.clone(), int_part, int_lit(0, span), i64_t, span);
                let mut pre = pre;
                let int = self
                    .hold(
                        Val {
                            node: int,
                            ty: Ty::Int,
                        },
                        &mut pre,
                    )
                    .node;
                let float_part = match (op, divisor.pow2) {
                    (BinOp::Percent(_), Some(n)) => self.float_mod_pow2(x.as_float(span), n, span),
                    _ => call(
                        ff,
                        vec![x.as_float(span), y.as_float(span)],
                        f64_t.clone(),
                        span,
                    ),
                };
                let float = float_word(is_int.clone(), int.clone(), float_part, span);
                // The call is below the value, where the check after a
                // call would not find it.
                let v = number(pre, int, float, is_int);
                if plain_divisor { v } else { self.guard(v) }
            }
            BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                let eq = self.number_compare_parts(&x, &y, Compare::Eq, span);
                let v = if matches!(op, BinOp::TildeEqual(_)) {
                    unary(UnaryOp::Not, eq, bool_t, span)
                } else {
                    eq
                };
                Val {
                    node: block_value(pre, v, span),
                    ty: Ty::Bool,
                }
            }
            BinOp::LessThan(_)
            | BinOp::LessThanEqual(_)
            | BinOp::GreaterThan(_)
            | BinOp::GreaterThanEqual(_) => {
                let (l, r, how) = match op {
                    BinOp::LessThan(_) => (&x, &y, Compare::Lt),
                    BinOp::LessThanEqual(_) => (&x, &y, Compare::Le),
                    BinOp::GreaterThan(_) => (&y, &x, Compare::Lt),
                    _ => (&y, &x, Compare::Le),
                };
                let v = self.number_compare_parts(l, r, how, span);
                Val {
                    node: block_value(pre, v, span),
                    ty: Ty::Bool,
                }
            }
            _ => unreachable!("the number path takes only arithmetic and comparison"),
        }
    }

    /// `a` against `b`, at least one of them a number of run-time kind.
    pub(super) fn number_compare(&mut self, a: Val, b: Val, how: Compare, span: Span) -> Node {
        let a = Val {
            node: self.coerce(a, Ty::Number),
            ty: Ty::Number,
        };
        let b = Val {
            node: self.coerce(b, Ty::Number),
            ty: Ty::Number,
        };
        let (mut pre, x) = self.number_parts(a);
        let (mut pre_b, y) = self.number_parts(b);
        pre.append(&mut pre_b);
        let v = self.number_compare_parts(&x, &y, how, span);
        block_value(pre, v, span)
    }

    /// Two integers compare as integers, two floats as floats, and an
    /// integer against a float exactly, as Lua does.
    pub(super) fn number_compare_parts(
        &mut self,
        x: &NumberParts,
        y: &NumberParts,
        how: Compare,
        span: Span,
    ) -> Node {
        let bool_t = prim(PrimitiveType::Bool);
        let op = match how {
            Compare::Eq => BinaryOp::Eq,
            Compare::Lt => BinaryOp::Lt,
            Compare::Le => BinaryOp::Le,
        };
        let ints = binary(op, x.int.clone(), y.int.clone(), bool_t.clone(), span);
        let floats = binary(op, x.float.clone(), y.float.clone(), bool_t.clone(), span);
        let x_int = mixed_numeric_compare(x.int.clone(), y.float.clone(), how, true, span);
        let y_int = mixed_numeric_compare(y.int.clone(), x.float.clone(), how, false, span);
        // Two numbers of one kind compare by a select; an integer
        // against a float takes the exact helper.
        let same = if_value(x.is_int(span), ints, floats, bool_t.clone(), span);
        let mixed = if_value(x.is_int(span), x_int, y_int, bool_t.clone(), span);
        let same_kind = binary(
            BinaryOp::Eq,
            x.tag.clone(),
            y.tag.clone(),
            bool_t.clone(),
            span,
        );
        if_value(same_kind, same, mixed, bool_t, span)
    }
}
