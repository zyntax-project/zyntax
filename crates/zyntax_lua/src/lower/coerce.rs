//! Coerce for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    pub(super) fn nil_val(&self, span: Span) -> Val {
        Val {
            node: nil(span),
            ty: Ty::Nil,
        }
    }

    /// `v` as a value of `target`.
    pub(super) fn coerce(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        let target = if target == Ty::Unknown {
            Ty::Nil
        } else {
            target
        };
        // A scalar that may only be nil besides holds its number in the
        // number's part; any other conversion is the scalar's.
        match (v.ty, target) {
            (Ty::IntOrNil, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                return block_value(pre, n.int, span);
            }
            (Ty::FloatOrNil, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                return block_value(pre, n.float, span);
            }
            // A dynamic value known to be the number or nil: nil, or
            // the payload.
            (Ty::Any, Ty::IntOrNil | Ty::FloatOrNil) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let number = if target == Ty::IntOrNil {
                    let payload = call(
                        "zb_box_payload_i64",
                        vec![b.node],
                        prim(PrimitiveType::I64),
                        span,
                    );
                    number_of_int(payload, span)
                } else {
                    let payload = call(
                        "zb_box_payload_f64",
                        vec![b.node],
                        prim(PrimitiveType::F64),
                        span,
                    );
                    number_of_float(payload, span)
                };
                let value = if_value(is_nil, scalar_of_nil(span), number, number_type(), span);
                return block_value(pre, value, span);
            }
            (Ty::IntOrNil | Ty::FloatOrNil, Ty::Any) => {
                let int = v.ty == Ty::IntOrNil;
                let (pre, n) = self.number_parts(v);
                let boxed = if int {
                    call("zb_box_i64", vec![n.int.clone()], Type::Any, span)
                } else {
                    call("zb_box_f64", vec![n.float.clone()], Type::Any, span)
                };
                let value = if_value(n.has_tag(TAG_NIL, span), nil(span), boxed, Type::Any, span);
                return block_value(pre, value, span);
            }
            _ => {}
        }
        let as_scalar = |t: Ty| if t.is_tagged() { Ty::Scalar } else { t };
        let v = Val {
            node: v.node,
            ty: as_scalar(v.ty),
        };
        let target = as_scalar(target);
        match (v.ty, target) {
            (a, b) if a == b => v.node,
            (Ty::Unknown, _) => v.node,
            (Ty::Nil, Ty::Any) => v.node,
            (Ty::Func(_), Ty::Any) | (Ty::Any, Ty::Func(_)) | (Ty::Func(_), Ty::Func(_)) => v.node,
            (Ty::Int, Ty::Float) => cast(v.node, prim(PrimitiveType::F64), span),
            (Ty::Int, Ty::Number) => number_of_int(v.node, span),
            (Ty::Float, Ty::Number) => number_of_float(v.node, span),
            (Ty::Number, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.as_float(span), span)
            }
            (Ty::Number, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    n.int,
                    cast(n.float, prim(PrimitiveType::I64), span),
                    prim(PrimitiveType::I64),
                    span,
                );
                block_value(pre, value, span)
            }
            (Ty::Number, Ty::Any) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    call("zb_box_i64", vec![n.int], Type::Any, span),
                    call("zb_box_f64", vec![n.float], Type::Any, span),
                    Type::Any,
                    span,
                );
                block_value(pre, value, span)
            }
            // A dynamic value known to hold a number: its kind is read
            // off the box with its payload.
            (Ty::Any, Ty::Number) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_int = binary(
                    BinaryOp::Ne,
                    call(
                        "zb_any_category",
                        vec![b.node.clone()],
                        prim(PrimitiveType::I64),
                        span,
                    ),
                    int_lit(library::FLOAT, span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let tag = tag_of_is_int(is_int, span);
                let int = call(
                    "zb_box_payload_i64",
                    vec![b.node.clone()],
                    prim(PrimitiveType::I64),
                    span,
                );
                let float = call(
                    "zb_box_payload_f64",
                    vec![b.node],
                    prim(PrimitiveType::F64),
                    span,
                );
                block_value(pre, number_value(tag, int, float, span), span)
            }
            // A number is a scalar as it is; the other scalars take
            // their tag.
            (Ty::Number, Ty::Scalar) | (Ty::Scalar, Ty::Number) => v.node,
            (Ty::Nil, Ty::Scalar) => {
                block_value(vec![expr_stmt(v.node)], scalar_of_nil(span), span)
            }
            (Ty::Bool, Ty::Scalar) => scalar_of_bool(v.node, span),
            (Ty::Int, Ty::Scalar) => number_of_int(v.node, span),
            (Ty::Float, Ty::Scalar) => number_of_float(v.node, span),
            (Ty::Scalar, Ty::Any) => {
                let (pre, n) = self.number_parts(v);
                let bool_t = prim(PrimitiveType::Bool);
                let is_true = n.has_tag(TAG_TRUE, span);
                let is_bool = binary(
                    BinaryOp::Lt,
                    n.tag.clone(),
                    int_lit(TAG_INT, span),
                    bool_t,
                    span,
                );
                let value = if_value(
                    n.has_tag(TAG_NIL, span),
                    nil(span),
                    if_value(
                        is_bool,
                        call("zb_box_bool", vec![is_true], Type::Any, span),
                        if_value(
                            n.has_tag(TAG_INT, span),
                            call("zb_box_i64", vec![n.int], Type::Any, span),
                            call("zb_box_f64", vec![n.float], Type::Any, span),
                            Type::Any,
                            span,
                        ),
                        Type::Any,
                        span,
                    ),
                    Type::Any,
                    span,
                );
                block_value(pre, value, span)
            }
            // A dynamic value known to be a scalar: its tag is read off
            // the box.
            (Ty::Any, Ty::Scalar) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let i64_t = prim(PrimitiveType::I64);
                let bool_t = prim(PrimitiveType::Bool);
                let category = call("zb_any_category", vec![b.node.clone()], i64_t.clone(), span);
                let is_cat = |c: i64| {
                    binary(
                        BinaryOp::Eq,
                        category.clone(),
                        int_lit(c, span),
                        bool_t.clone(),
                        span,
                    )
                };
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    bool_t.clone(),
                    span,
                );
                let truth = binary(
                    BinaryOp::Ne,
                    call(
                        "zb_box_payload_bool",
                        vec![b.node.clone()],
                        prim(PrimitiveType::I32),
                        span,
                    ),
                    int32_lit(0, span),
                    bool_t.clone(),
                    span,
                );
                let as_bool = scalar_of_bool(truth, span);
                let as_float = number_of_float(
                    call(
                        "zb_box_payload_f64",
                        vec![b.node.clone()],
                        prim(PrimitiveType::F64),
                        span,
                    ),
                    span,
                );
                let as_int =
                    number_of_int(call("zb_box_payload_i64", vec![b.node], i64_t, span), span);
                let value = if_value(
                    is_nil,
                    scalar_of_nil(span),
                    if_value(
                        is_cat(library::BOOL),
                        as_bool,
                        if_value(
                            is_cat(library::FLOAT),
                            as_float,
                            as_int,
                            number_type(),
                            span,
                        ),
                        number_type(),
                        span,
                    ),
                    number_type(),
                    span,
                );
                block_value(pre, value, span)
            }
            // A scalar the types know the kind of.
            (Ty::Scalar, Ty::Int) => {
                let (pre, n) = self.number_parts(v);
                let value = if_value(
                    n.is_int(span),
                    n.int,
                    cast(n.float, prim(PrimitiveType::I64), span),
                    prim(PrimitiveType::I64),
                    span,
                );
                block_value(pre, value, span)
            }
            (Ty::Scalar, Ty::Float) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.as_float(span), span)
            }
            (Ty::Scalar, Ty::Bool) => {
                let (pre, n) = self.number_parts(v);
                block_value(pre, n.has_tag(TAG_TRUE, span), span)
            }
            (Ty::Scalar, Ty::Nil) => block_value(vec![expr_stmt(v.node)], nil(span), span),
            // A shaped table is a table; null is nil. Only a table slot
            // takes a shaped value as a table, and it keeps nil as null
            // with its present bit clear.
            (Ty::Table, Ty::Shape(_))
            | (Ty::Shape(_), Ty::Shape(_))
            | (Ty::Shape(_), Ty::Table) => v.node,
            (Ty::Nil, Ty::Shape(_)) => block_value(
                vec![expr_stmt(v.node)],
                null(self.ir(Ty::Table), span),
                span,
            ),
            (Ty::Shape(_), Ty::Any) => {
                let mut pre = Vec::new();
                let t = self.hold(v, &mut pre);
                let is_null = binary(
                    BinaryOp::Eq,
                    t.node.clone(),
                    null(self.ir(Ty::Table), span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let boxed = self.box_table(t.node);
                block_value(
                    pre,
                    if_value(is_null, nil(span), boxed, Type::Any, span),
                    span,
                )
            }
            (Ty::Any, Ty::Shape(_)) => {
                let mut pre = Vec::new();
                let b = self.hold(v, &mut pre);
                let is_nil = binary(
                    BinaryOp::Eq,
                    b.node.clone(),
                    nil(span),
                    prim(PrimitiveType::Bool),
                    span,
                );
                let table_t = self.ir(Ty::Table);
                let unboxed = self.unbox_table(b.node);
                block_value(
                    pre,
                    if_value(is_nil, null(table_t.clone(), span), unboxed, table_t, span),
                    span,
                )
            }
            (Ty::Shape(_), Ty::Nil) => block_value(vec![expr_stmt(v.node)], nil(span), span),
            (Ty::Bool, Ty::Any) => call("zb_box_bool", vec![v.node], Type::Any, span),
            (Ty::Int, Ty::Any) => call("zb_box_i64", vec![v.node], Type::Any, span),
            (Ty::Float, Ty::Any) => call("zb_box_f64", vec![v.node], Type::Any, span),
            (Ty::Str, Ty::Any) => call("zb_box_str", vec![v.node], Type::Any, span),
            (Ty::Table, Ty::Any) => self.box_table(v.node),
            // A dynamic value known to hold the type: read straight out.
            (Ty::Any, Ty::Bool) => binary(
                BinaryOp::Ne,
                call(
                    "zb_box_payload_bool",
                    vec![v.node],
                    prim(PrimitiveType::I32),
                    span,
                ),
                int32_lit(0, span),
                prim(PrimitiveType::Bool),
                span,
            ),
            (Ty::Any, Ty::Int) => call(
                "zb_box_payload_i64",
                vec![v.node],
                prim(PrimitiveType::I64),
                span,
            ),
            (Ty::Any, Ty::Float) => call(
                "zb_box_payload_f64",
                vec![v.node],
                prim(PrimitiveType::F64),
                span,
            ),
            (Ty::Any, Ty::Str) => call(
                "zb_box_get_str",
                vec![v.node],
                prim(PrimitiveType::String),
                span,
            ),
            (Ty::Any, Ty::Table) => self.unbox_table(v.node),
            (Ty::Any, Ty::Nil) => v.node,
            // A nil where a typed value was expected: the value's zero,
            // after the expression ran. Inference joins nil into the
            // type, so this is only reached for a slot never read.
            (Ty::Nil, t) => {
                let zero = match t {
                    Ty::Int => int_lit(0, span),
                    Ty::Float => float_lit(0.0, span),
                    Ty::Number => number_of_int(int_lit(0, span), span),
                    Ty::Scalar => scalar_of_nil(span),
                    Ty::Bool => bool_lit(false, span),
                    Ty::Str => str_lit("", span),
                    Ty::Table | Ty::Shape(_) => null(self.ir(Ty::Table), span),
                    _ => nil(span),
                };
                block_value(vec![expr_stmt(v.node)], zero, span)
            }
            (Ty::Float, Ty::Int) => cast(v.node, prim(PrimitiveType::I64), span),
            (a, b) => {
                // Anything else goes through the box.
                let boxed = self.coerce(v, Ty::Any);
                if b == Ty::Any {
                    return boxed;
                }
                let _ = a;
                self.coerce(
                    Val {
                        node: boxed,
                        ty: Ty::Any,
                    },
                    b,
                )
            }
        }
    }

    pub(super) fn boxed(&mut self, v: Val) -> Node {
        self.coerce(v, Ty::Any)
    }

    pub(super) fn box_table(&mut self, t: Node) -> Node {
        let span = t.span;
        call(
            "zb_box_instance",
            vec![
                cast(t, prim(PrimitiveType::I64), span),
                int32_lit(library::table_tag() as i32, span),
            ],
            Type::Any,
            span,
        )
    }

    pub(super) fn unbox_table(&mut self, x: Node) -> Node {
        let span = x.span;
        cast(
            call(
                "zb_unbox_instance_raw",
                vec![x],
                prim(PrimitiveType::I64),
                span,
            ),
            self.ir(Ty::Table),
            span,
        )
    }

    /// Whether the node is a plain read, safe to repeat.
    pub(super) fn is_simple(node: &Node) -> bool {
        matches!(
            node.node,
            TypedExpression::Variable(_) | TypedExpression::Literal(_)
        )
    }

    /// The value copied into a temporary, unless it is a literal: a
    /// variable read now, before a statement that may assign it.
    pub(super) fn snapshot(&mut self, v: Val, pre: &mut Vec<St>) -> Val {
        if matches!(v.node.node, TypedExpression::Literal(_)) {
            return v;
        }
        let span = v.node.span;
        let name = self.temp();
        let ty = self.ir(v.ty);
        if let Some(fields) = self.value_fields(&v) {
            self.field_values.insert(name, fields);
        }
        pre.push(let_(name, ty.clone(), v.node, span));
        Val {
            node: var(name, ty, span),
            ty: v.ty,
        }
    }

    /// The value held in a temporary, so it can be read more than
    /// once: the statement binding it, and the read.
    pub(super) fn hold(&mut self, v: Val, pre: &mut Vec<St>) -> Val {
        if Self::is_simple(&v.node) {
            return v;
        }
        let span = v.node.span;
        let name = self.temp();
        let ty = self.ir(v.ty);
        if let Some(fields) = self.value_fields(&v) {
            self.field_values.insert(name, fields);
        }
        pre.push(let_(name, ty.clone(), v.node, span));
        Val {
            node: var(name, ty, span),
            ty: v.ty,
        }
    }

    /// `x % n` for a float `x` and a positive power of two `n`: the
    /// quotient, its floor and the product are all exact, and so is the
    /// difference; a zero keeps `x`'s sign, as fmod's does.
    pub(super) fn float_mod_pow2(&mut self, x: Node, n: i64, span: Span) -> Node {
        let f64_t = prim(PrimitiveType::F64);
        let mut pre = Vec::new();
        let x = self
            .hold(
                Val {
                    node: x,
                    ty: Ty::Float,
                },
                &mut pre,
            )
            .node;
        let quotient = binary(
            BinaryOp::Mul,
            x.clone(),
            float_lit(1.0 / n as f64, span),
            f64_t.clone(),
            span,
        );
        let floored = call("floor", vec![quotient], f64_t.clone(), span);
        let product = binary(
            BinaryOp::Mul,
            floored,
            float_lit(n as f64, span),
            f64_t.clone(),
            span,
        );
        let r = self
            .hold(
                Val {
                    node: binary(BinaryOp::Sub, x.clone(), product, f64_t.clone(), span),
                    ty: Ty::Float,
                },
                &mut pre,
            )
            .node;
        let zero = binary(
            BinaryOp::Eq,
            r.clone(),
            float_lit(0.0, span),
            prim(PrimitiveType::Bool),
            span,
        );
        let signed_zero = binary(BinaryOp::Mul, x, float_lit(0.0, span), f64_t.clone(), span);
        block_value(pre, if_value(zero, signed_zero, r, f64_t, span), span)
    }
}
