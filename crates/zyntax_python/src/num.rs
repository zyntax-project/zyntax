//! Run-time numbers: a local that holds more than one of int, float,
//! bool and None, carried unboxed as [`Ty::Num`].
//!
//! A Num is the value struct `(tag: i64, int: i64, float: f64)` with the
//! tags Lua's scalars use: NONE 0, FALSE 1, TRUE 2, INT 3, FLOAT 4.
//! Invariant: FALSE and TRUE have int 0 and 1; NONE has int 0 and float
//! 0.0; INT's int is the value and its float unspecified; FLOAT's float
//! is the value and its int unspecified. Every reader selects by the tag,
//! so a Num is held before its parts are read twice.
//!
//! A reader without an arm of its own reads a Num as the box a plain
//! value of its kind would be (`zb_num_box`), which the dynamic layer
//! already handles, so a site either computes on the parts or behaves as
//! it did for the boxed value.

use super::{Lowerer, Node, Stmt, Val, binary, call, cast, int_lit, node, slot};
use crate::types::Ty;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{TypedExpression, TypedIfExpr, TypedLiteral};
use zyntax_typed_ast::{BinaryOp, Type, TypedNode};

pub(crate) const TAG_NONE: i64 = 0;
pub(crate) const TAG_FALSE: i64 = 1;
pub(crate) const TAG_TRUE: i64 = 2;
pub(crate) const TAG_INT: i64 = 3;
pub(crate) const TAG_FLOAT: i64 = 4;

/// Whether locals may be [`Ty::Num`]. `ZYNTAX_DISABLE_NUM=1` makes such
/// a local dynamic, as it was before Num existed; safe to run with.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_NUM").is_none())
}

/// The kinds that are numbers: a Num of only these never raises the
/// TypeError None would in arithmetic.
const NUMBERS: u8 = Ty::NUM_BOOL | Ty::NUM_INT | Ty::NUM_FLOAT;

/// What `left op right` is when a [`Ty::Num`] takes part and the
/// lowering computes it on the parts: `+ - * // %` and `** 2` on
/// operands that cannot be None give an int when neither may be a float,
/// a float when one side is one, and otherwise the run-time number; `/`
/// gives a float. Any other operation reads the Num as its box.
pub(crate) fn num_binop(
    op: ruff_python_ast::Operator,
    l: Ty,
    r: Ty,
    right: &ruff_python_ast::Expr,
) -> Option<Ty> {
    use ruff_python_ast::Operator as O;
    if !matches!(l, Ty::Num(_)) && !matches!(r, Ty::Num(_)) {
        return None;
    }
    let squared = op == O::Pow && is_two(right);
    if !matches!(
        op,
        O::Add | O::Sub | O::Mult | O::Div | O::FloorDiv | O::Mod
    ) && !squared
    {
        return None;
    }
    let (ml, mr) = (l.mask()?, r.mask()?);
    if (ml | mr) & !NUMBERS != 0 {
        return None;
    }
    Some(if op == O::Div {
        Ty::Float
    } else if (ml | mr) & Ty::NUM_FLOAT == 0 {
        Ty::Int
    } else if ml == Ty::NUM_FLOAT || mr == Ty::NUM_FLOAT {
        Ty::Float
    } else {
        Ty::Num(Ty::NUM_INT | Ty::NUM_FLOAT)
    })
}

/// What `-x` and `+x` are for a [`Ty::Num`] `x` that cannot be None: a
/// bool becomes an int, an int or a float stays itself.
pub(crate) fn num_unary(t: Ty) -> Option<Ty> {
    let m = t.mask()?;
    if m & !NUMBERS != 0 {
        return None;
    }
    let bool_as_int = if m & Ty::NUM_BOOL != 0 {
        Ty::NUM_INT
    } else {
        0
    };
    Some(Ty::num((m & !Ty::NUM_BOOL) | bool_as_int))
}

/// Whether `e` is the int literal 2.
fn is_two(e: &ruff_python_ast::Expr) -> bool {
    matches!(e, ruff_python_ast::Expr::NumberLiteral(n)
        if matches!(&n.value, ruff_python_ast::Number::Int(i) if i.as_u8() == Some(2)))
}

fn float_lit(v: f64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Float(v)),
        Ty::Float,
        span,
    )
}

fn select(c: Node, t: Node, e: Node, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::If(TypedIfExpr {
            condition: Box::new(c),
            then_branch: Box::new(t),
            else_branch: Box::new(e),
        }),
        ty,
        span,
    )
}

fn value(tag: Node, int: Node, float: Node, ty: Ty, span: Span) -> Node {
    node(TypedExpression::Tuple(vec![tag, int, float]), ty, span)
}

/// The parts of a held Num: plain reads.
pub(crate) struct NumParts {
    pub(crate) tag: Node,
    pub(crate) int: Node,
    pub(crate) float: Node,
}

impl NumParts {
    fn of(held: &Node, span: Span) -> NumParts {
        NumParts {
            tag: slot(held.clone(), 0, Ty::Int, span),
            int: slot(held.clone(), 1, Ty::Int, span),
            float: slot(held.clone(), 2, Ty::Float, span),
        }
    }

    fn has_tag(&self, tag: i64, span: Span) -> Node {
        binary(
            BinaryOp::Eq,
            self.tag.clone(),
            int_lit(tag, span),
            Ty::Bool,
            span,
        )
    }

    /// The value as a float: the float, or the int (a bool's 0 or 1)
    /// converted.
    fn as_f64(&self, span: Span) -> Node {
        select(
            self.has_tag(TAG_FLOAT, span),
            self.float.clone(),
            cast(self.int.clone(), Ty::Float, span),
            Ty::Float,
            span,
        )
    }
}

impl Lowerer<'_> {
    /// `v` held where its parts can be read: a variable as it is,
    /// anything else bound to a temporary ahead of the reads.
    fn num_held(&mut self, v: Val, pre: &mut Vec<Stmt>, span: Span) -> NumParts {
        let held = if matches!(v.node.node, TypedExpression::Variable(_)) {
            v.node
        } else {
            self.hold(v, pre, span).node
        };
        NumParts::of(&held, span)
    }

    fn with_pre(pre: Vec<Stmt>, value: Node, ty: Ty, span: Span) -> Node {
        if pre.is_empty() {
            value
        } else {
            Self::block_value(pre, value, ty, span)
        }
    }

    /// A plain value of int, float, bool or None as the Num `target`.
    pub(crate) fn coerce_to_num(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Int => value(
                int_lit(TAG_INT, span),
                v.node,
                float_lit(0.0, span),
                target,
                span,
            ),
            Ty::Float => value(
                int_lit(TAG_FLOAT, span),
                int_lit(0, span),
                v.node,
                target,
                span,
            ),
            Ty::Bool => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre, span).node;
                let tag = select(
                    b.clone(),
                    int_lit(TAG_TRUE, span),
                    int_lit(TAG_FALSE, span),
                    Ty::Int,
                    span,
                );
                let int = cast(b, Ty::Int, span);
                let built = value(tag, int, float_lit(0.0, span), target, span);
                Self::with_pre(pre, built, target, span)
            }
            Ty::None => {
                let none = value(
                    int_lit(TAG_NONE, span),
                    int_lit(0, span),
                    float_lit(0.0, span),
                    target,
                    span,
                );
                if matches!(
                    v.node.node,
                    TypedExpression::Literal(_) | TypedExpression::Variable(_)
                ) {
                    none
                } else {
                    let effect = TypedNode::new(
                        zyntax_typed_ast::typed_ast::TypedStatement::Expression(Box::new(v.node)),
                        Type::Unknown,
                        span,
                    );
                    Self::block_value(vec![effect], none, target, span)
                }
            }
            // One Num into a wider one is the same struct.
            Ty::Num(_) => {
                let mut n = v.node;
                n.ty = super::ir(target);
                n
            }
            _ => {
                // Anything else is read out of its box, checked against
                // the kinds `target` admits.
                let boxed = self.coerce(v, Ty::Object);
                self.num_of_box(boxed, target, span)
            }
        }
    }

    /// A box read as the Num `target`: a TypeError when it holds a kind
    /// the Num does not admit.
    fn num_of_box(&mut self, boxed: Node, target: Ty, span: Span) -> Node {
        let mask = target.mask().unwrap_or(0);
        let mut pre = Vec::new();
        let held = self
            .hold(
                Val {
                    node: boxed,
                    ty: Ty::Object,
                },
                &mut pre,
                span,
            )
            .node;
        let tag = call(
            "zb_num_tag_in",
            vec![held.clone(), int_lit(i64::from(mask), span)],
            Ty::Int,
            span,
        );
        let tag = self
            .guard(
                Val {
                    node: tag,
                    ty: Ty::Int,
                },
                span,
            )
            .node;
        pre.append(&mut self.hoisted);
        let built = value(
            tag,
            call("zb_num_int", vec![held.clone()], Ty::Int, span),
            call("zb_num_float", vec![held], Ty::Float, span),
            target,
            span,
        );
        Self::with_pre(pre, built, target, span)
    }

    /// A Num as `target`: its box, a float or an int where every kind it
    /// admits converts without a raise, and otherwise through its box.
    pub(crate) fn coerce_from_num(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        let mask = v.ty.mask().unwrap_or(0);
        let mut pre = Vec::new();
        let result = match target {
            Ty::Object => {
                let p = self.num_held(v, &mut pre, span);
                call("zb_num_box", vec![p.tag, p.int, p.float], Ty::Object, span)
            }
            Ty::Float if mask & !NUMBERS == 0 => self.num_held(v, &mut pre, span).as_f64(span),
            Ty::Int if mask & !(Ty::NUM_BOOL | Ty::NUM_INT) == 0 => {
                self.num_held(v, &mut pre, span).int
            }
            Ty::Num(_) => return self.coerce_to_num(v, target),
            _ => {
                let boxed = self.coerce_from_num(v, Ty::Object);
                return self.coerce(
                    Val {
                        node: boxed,
                        ty: Ty::Object,
                    },
                    target,
                );
            }
        };
        Self::with_pre(pre, result, target, span)
    }

    /// `v`, read as its box when it is a Num or a list or None.
    pub(crate) fn boxed_num(&mut self, v: Val) -> Val {
        if !matches!(v.ty, Ty::Num(_) | Ty::MaybeList(_)) {
            return v;
        }
        let node = self.coerce(v, Ty::Object);
        Val {
            node,
            ty: Ty::Object,
        }
    }

    /// `bool(v)` of a Num: True, a nonzero int or a nonzero float (NaN
    /// is true, -0.0 false); None and False are false.
    pub(crate) fn num_truthy(&mut self, v: Val) -> Node {
        let span = v.node.span;
        let mut pre = Vec::new();
        let p = self.num_held(v, &mut pre, span);
        // NONE's and FALSE's int is 0 and TRUE's 1, so up to INT the
        // value is true when its int is nonzero; FLOAT's int is unspecified.
        let int_true = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Le,
                p.tag.clone(),
                int_lit(TAG_INT, span),
                Ty::Bool,
                span,
            ),
            binary(
                BinaryOp::Ne,
                p.int.clone(),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            Ty::Bool,
            span,
        );
        let float_true = binary(
            BinaryOp::And,
            p.has_tag(TAG_FLOAT, span),
            binary(
                BinaryOp::Ne,
                p.float.clone(),
                float_lit(0.0, span),
                Ty::Bool,
                span,
            ),
            Ty::Bool,
            span,
        );
        let truth = binary(BinaryOp::Or, int_true, float_true, Ty::Bool, span);
        Self::with_pre(pre, truth, Ty::Bool, span)
    }

    /// `v is None` of a Num: its tag.
    pub(crate) fn num_is_none(&mut self, v: Val) -> Node {
        let span = v.node.span;
        let mut pre = Vec::new();
        let p = self.num_held(v, &mut pre, span);
        let test = p.has_tag(TAG_NONE, span);
        Self::with_pre(pre, test, Ty::Bool, span)
    }

    /// `left op right` as [`num_binop`] types it, on the parts. An
    /// operation whose operands have one kind each is the typed one on
    /// those kinds.
    pub(crate) fn num_arith(
        &mut self,
        op: ruff_python_ast::Operator,
        left: Val,
        right: Val,
        right_expr: &ruff_python_ast::Expr,
        ty: Ty,
        span: Span,
    ) -> crate::Result<Val> {
        use ruff_python_ast::Operator as O;
        let ints = |t: Ty| t.mask().unwrap_or(0) & !(Ty::NUM_BOOL | Ty::NUM_INT) == 0;
        let floats = |t: Ty| t.mask() == Some(Ty::NUM_FLOAT);
        let kinds = if ints(left.ty) && ints(right.ty) {
            Some(Ty::Int)
        } else if floats(left.ty) || floats(right.ty) {
            Some(Ty::Float)
        } else {
            None
        };
        if let Some(kind) = kinds {
            let l = self.coerce(left, kind);
            let r = self.coerce(right, kind);
            return self.arithmetic(
                op,
                Val { node: l, ty: kind },
                Val { node: r, ty: kind },
                right_expr,
                span,
            );
        }
        let mut pre = Vec::new();
        if op == O::Pow {
            // `x ** 2` is `x * x` on each path.
            let x = self.hold(left, &mut pre, span);
            self.hoisted.append(&mut pre);
            return self.num_arith(O::Mult, x.clone(), x, right_expr, ty, span);
        }
        let wide = |t: Ty| Ty::Num(t.mask().unwrap_or(0) | Ty::NUM_INT | Ty::NUM_FLOAT);
        let (lty, rty) = (wide(left.ty), wide(right.ty));
        let l = Val {
            node: self.coerce(left, lty),
            ty: lty,
        };
        let r = Val {
            node: self.coerce(right, rty),
            ty: rty,
        };
        let l = self.num_held(l, &mut pre, span);
        let r = self.num_held(r, &mut pre, span);
        let is_int = |p: &NumParts| {
            binary(
                BinaryOp::Le,
                p.tag.clone(),
                int_lit(TAG_INT, span),
                Ty::Bool,
                span,
            )
        };
        let both_int = binary(BinaryOp::And, is_int(&l), is_int(&r), Ty::Bool, span);
        let node = match op {
            // Both paths are free of traps, so both are computed and the
            // tags pick one.
            O::Add | O::Sub | O::Mult => {
                let bin = match op {
                    O::Add => BinaryOp::Add,
                    O::Sub => BinaryOp::Sub,
                    _ => BinaryOp::Mul,
                };
                let tag = select(
                    both_int,
                    int_lit(TAG_INT, span),
                    int_lit(TAG_FLOAT, span),
                    Ty::Int,
                    span,
                );
                let int = binary(bin, l.int.clone(), r.int.clone(), Ty::Int, span);
                let float = binary(bin, l.as_f64(span), r.as_f64(span), Ty::Float, span);
                value(tag, int, float, ty, span)
            }
            // A float quotient, whose zero text depends on the kinds.
            O::Div => {
                let divisor = self.hold(
                    Val {
                        node: r.as_f64(span),
                        ty: Ty::Float,
                    },
                    &mut pre,
                    span,
                );
                let mut raise_int = Vec::new();
                self.raise_named(
                    "ZeroDivisionError",
                    super::str_lit("division by zero", span),
                    span,
                    &mut raise_int,
                );
                let mut raise_float = Vec::new();
                self.raise_named(
                    "ZeroDivisionError",
                    super::str_lit("float division by zero", span),
                    span,
                    &mut raise_float,
                );
                let raise = vec![if_stmt(both_int, raise_int, Some(raise_float), span)];
                pre.push(if_stmt(
                    binary(
                        BinaryOp::Eq,
                        divisor.node.clone(),
                        float_lit(0.0, span),
                        Ty::Bool,
                        span,
                    ),
                    raise,
                    None,
                    span,
                ));
                binary(BinaryOp::Div, l.as_f64(span), divisor.node, Ty::Float, span)
            }
            // `//` and `%` branch on the kinds: the int path traps on
            // operands a float's words would give it.
            _ => {
                let saved = std::mem::take(&mut self.hoisted);
                let int = self.arithmetic(
                    op,
                    Val {
                        node: l.int.clone(),
                        ty: Ty::Int,
                    },
                    Val {
                        node: r.int.clone(),
                        ty: Ty::Int,
                    },
                    right_expr,
                    span,
                )?;
                let int_pre = std::mem::take(&mut self.hoisted);
                let float = self.arithmetic(
                    op,
                    Val {
                        node: l.as_f64(span),
                        ty: Ty::Float,
                    },
                    Val {
                        node: r.as_f64(span),
                        ty: Ty::Float,
                    },
                    right_expr,
                    span,
                )?;
                let float_pre = std::mem::replace(&mut self.hoisted, saved);
                let int = value(
                    int_lit(TAG_INT, span),
                    int.node,
                    float_lit(0.0, span),
                    ty,
                    span,
                );
                let float = value(
                    int_lit(TAG_FLOAT, span),
                    int_lit(0, span),
                    float.node,
                    ty,
                    span,
                );
                self.conditional_value(
                    both_int,
                    (int_pre, int),
                    (float_pre, float),
                    ty,
                    span,
                    &mut pre,
                )
            }
        };
        Ok(Val {
            node: Self::with_pre(pre, node, ty, span),
            ty,
        })
    }
}

impl Lowerer<'_> {
    /// `-x` or `+x` of a Num as [`num_unary`] types it, on the parts: the
    /// tag is FLOAT or INT, and both words follow.
    pub(crate) fn num_negate(&mut self, v: Val, negate: bool, ty: Ty, span: Span) -> Node {
        if ty == Ty::Int {
            let n = self.coerce(v, Ty::Int);
            return if negate {
                binary(BinaryOp::Sub, int_lit(0, span), n, Ty::Int, span)
            } else {
                n
            };
        }
        let mut pre = Vec::new();
        let p = self.num_held(v, &mut pre, span);
        let tag = select(
            p.has_tag(TAG_FLOAT, span),
            int_lit(TAG_FLOAT, span),
            int_lit(TAG_INT, span),
            Ty::Int,
            span,
        );
        let (int, float) = if negate {
            (
                binary(BinaryOp::Sub, int_lit(0, span), p.int, Ty::Int, span),
                node(
                    TypedExpression::Unary(zyntax_typed_ast::typed_ast::TypedUnary {
                        op: zyntax_typed_ast::UnaryOp::Minus,
                        operand: Box::new(p.float),
                    }),
                    Ty::Float,
                    span,
                ),
            )
        } else {
            (p.int, p.float)
        };
        let built = value(tag, int, float, ty, span);
        Self::with_pre(pre, built, ty, span)
    }

    /// `left op right` for an ordering or equality where a Num takes part
    /// and neither side may be None, on the parts: two ints compare as
    /// ints, anything else as floats, and an int beyond 2^53 against a
    /// float exactly. None when the comparison is not one of these.
    pub(crate) fn num_compare(
        &mut self,
        op: ruff_python_ast::CmpOp,
        left: Val,
        right: Val,
        span: Span,
    ) -> Option<Node> {
        use ruff_python_ast::CmpOp as C;
        if !matches!(left.ty, Ty::Num(_)) && !matches!(right.ty, Ty::Num(_)) {
            return None;
        }
        let bin = match op {
            C::Eq => BinaryOp::Eq,
            C::NotEq => BinaryOp::Ne,
            C::Lt => BinaryOp::Lt,
            C::LtE => BinaryOp::Le,
            C::Gt => BinaryOp::Gt,
            C::GtE => BinaryOp::Ge,
            _ => return None,
        };
        let (ml, mr) = (left.ty.mask()?, right.ty.mask()?);
        if (ml | mr) & !NUMBERS != 0 {
            return None;
        }
        let ints = |m: u8| m & !(Ty::NUM_BOOL | Ty::NUM_INT) == 0;
        let mut pre = Vec::new();
        // One kind a side: the typed comparison of those kinds.
        if (ints(ml) || ml == Ty::NUM_FLOAT) && (ints(mr) || mr == Ty::NUM_FLOAT) {
            let lk = if ints(ml) { Ty::Int } else { Ty::Float };
            let rk = if ints(mr) { Ty::Int } else { Ty::Float };
            let l = self.coerce(left, lk);
            let r = self.coerce(right, rk);
            return self
                .compare_one(op, Val { node: l, ty: lk }, Val { node: r, ty: rk }, span)
                .ok();
        }
        let wide = |t: Ty| Ty::Num(t.mask().unwrap_or(0) | Ty::NUM_INT | Ty::NUM_FLOAT);
        let (lty, rty) = (wide(left.ty), wide(right.ty));
        let l = Val {
            node: self.coerce(left, lty),
            ty: lty,
        };
        let r = Val {
            node: self.coerce(right, rty),
            ty: rty,
        };
        let l = self.num_held(l, &mut pre, span);
        let r = self.num_held(r, &mut pre, span);
        let is_int = |p: &NumParts| {
            binary(
                BinaryOp::Le,
                p.tag.clone(),
                int_lit(TAG_INT, span),
                Ty::Bool,
                span,
            )
        };
        let both_int = binary(BinaryOp::And, is_int(&l), is_int(&r), Ty::Bool, span);
        let plain = select(
            both_int.clone(),
            binary(bin, l.int.clone(), r.int.clone(), Ty::Bool, span),
            binary(bin, l.as_f64(span), r.as_f64(span), Ty::Bool, span),
            Ty::Bool,
            span,
        );
        // An int converts to float exactly within 2^53, and two ints
        // never convert; otherwise one side is a wide int, the other a
        // float.
        let wide_int = |p: &NumParts| {
            let limit = 1i64 << 53;
            binary(
                BinaryOp::And,
                is_int(p),
                binary(
                    BinaryOp::Or,
                    binary(
                        BinaryOp::Gt,
                        p.int.clone(),
                        int_lit(limit, span),
                        Ty::Bool,
                        span,
                    ),
                    binary(
                        BinaryOp::Lt,
                        p.int.clone(),
                        int_lit(-limit, span),
                        Ty::Bool,
                        span,
                    ),
                    Ty::Bool,
                    span,
                ),
                Ty::Bool,
                span,
            )
        };
        let inexact = binary(
            BinaryOp::And,
            binary(BinaryOp::Or, wide_int(&l), wide_int(&r), Ty::Bool, span),
            not(both_int, span),
            Ty::Bool,
            span,
        );
        let saved = std::mem::take(&mut self.hoisted);
        let int_left = self.exact_int_float_compare(
            op,
            Val {
                node: l.int.clone(),
                ty: Ty::Int,
            },
            Val {
                node: r.float.clone(),
                ty: Ty::Float,
            },
            span,
        )?;
        let int_left_pre = std::mem::take(&mut self.hoisted);
        let int_right = self.exact_int_float_compare(
            op,
            Val {
                node: l.float.clone(),
                ty: Ty::Float,
            },
            Val {
                node: r.int.clone(),
                ty: Ty::Int,
            },
            span,
        )?;
        let int_right_pre = std::mem::replace(&mut self.hoisted, saved);
        let mut exact_pre = Vec::new();
        let exact = self.conditional_value(
            is_int(&l),
            (int_left_pre, int_left),
            (int_right_pre, int_right),
            Ty::Bool,
            span,
            &mut exact_pre,
        );
        let n = self.conditional_value(
            inexact,
            (exact_pre, exact),
            (Vec::new(), plain),
            Ty::Bool,
            span,
            &mut pre,
        );
        Some(Self::with_pre(pre, n, Ty::Bool, span))
    }
}

fn not(n: Node, span: Span) -> Node {
    node(
        TypedExpression::Unary(zyntax_typed_ast::typed_ast::TypedUnary {
            op: zyntax_typed_ast::UnaryOp::Not,
            operand: Box::new(n),
        }),
        Ty::Bool,
        span,
    )
}

fn if_stmt(cond: Node, then: Vec<Stmt>, otherwise: Option<Vec<Stmt>>, span: Span) -> Stmt {
    use zyntax_typed_ast::typed_ast::{TypedBlock, TypedIf, TypedStatement};
    TypedNode::new(
        TypedStatement::If(TypedIf {
            condition: Box::new(cond),
            then_block: TypedBlock {
                statements: then,
                span,
            },
            else_block: otherwise.map(|statements| TypedBlock { statements, span }),
            span,
        }),
        Type::Unknown,
        span,
    )
}
