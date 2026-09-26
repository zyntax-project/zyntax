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
/// lowering computes it on the parts: `+ - *` on operands that cannot
/// be None. An int and a bool give an int, a float on either side a
/// float, otherwise the run-time number. Any other operation reads the
/// Num as its box.
pub(crate) fn num_binop(op: ruff_python_ast::Operator, l: Ty, r: Ty) -> Option<Ty> {
    use ruff_python_ast::Operator as O;
    if !matches!(l, Ty::Num(_)) && !matches!(r, Ty::Num(_)) {
        return None;
    }
    if !matches!(op, O::Add | O::Sub | O::Mult) {
        return None;
    }
    let (ml, mr) = (l.mask()?, r.mask()?);
    if (ml | mr) & !NUMBERS != 0 {
        return None;
    }
    Some(if (ml | mr) & Ty::NUM_FLOAT == 0 {
        Ty::Int
    } else if ml == Ty::NUM_FLOAT || mr == Ty::NUM_FLOAT {
        Ty::Float
    } else {
        Ty::Num(Ty::NUM_INT | Ty::NUM_FLOAT)
    })
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

    /// `v`, read as its box when it is a Num.
    pub(crate) fn boxed_num(&mut self, v: Val) -> Val {
        if !matches!(v.ty, Ty::Num(_)) {
            return v;
        }
        let node = self.coerce_from_num(v, Ty::Object);
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

    /// `left op right` as [`num_binop`] types it, on the parts.
    pub(crate) fn num_arith(
        &mut self,
        op: ruff_python_ast::Operator,
        left: Val,
        right: Val,
        ty: Ty,
        span: Span,
    ) -> Node {
        use ruff_python_ast::Operator as O;
        let bin = match op {
            O::Add => BinaryOp::Add,
            O::Sub => BinaryOp::Sub,
            _ => BinaryOp::Mul,
        };
        if matches!(ty, Ty::Float | Ty::Int) {
            let l = self.coerce(left, ty);
            let r = self.coerce(right, ty);
            return binary(bin, l, r, ty, span);
        }
        // Both paths are free of traps, so both are computed and the
        // tags pick one.
        let mut pre = Vec::new();
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
        let both_int = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Le,
                l.tag.clone(),
                int_lit(TAG_INT, span),
                Ty::Bool,
                span,
            ),
            binary(
                BinaryOp::Le,
                r.tag.clone(),
                int_lit(TAG_INT, span),
                Ty::Bool,
                span,
            ),
            Ty::Bool,
            span,
        );
        let tag = select(
            both_int,
            int_lit(TAG_INT, span),
            int_lit(TAG_FLOAT, span),
            Ty::Int,
            span,
        );
        let int = binary(bin, l.int.clone(), r.int.clone(), Ty::Int, span);
        let float = binary(bin, l.as_f64(span), r.as_f64(span), Ty::Float, span);
        let built = value(tag, int, float, ty, span);
        Self::with_pre(pre, built, ty, span)
    }
}
