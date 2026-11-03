//! Expressions for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// An expression in single-value position.
    pub(super) fn expr(&mut self, e: &Expression) -> Result<Val> {
        let span = span_of(e);
        match e {
            Expression::Number(t) => {
                let text = t.token().to_string();
                Ok(match crate::host::parse_numeral(text.trim()) {
                    crate::host::Numeral::Int(v) => Val {
                        node: int_lit(v, span),
                        ty: Ty::Int,
                    },
                    crate::host::Numeral::Float(v) => Val {
                        node: float_lit(v, span),
                        ty: Ty::Float,
                    },
                    crate::host::Numeral::None => {
                        return Err(Error::Syntax {
                            message: format!("malformed number near '{}'", text.trim()),
                            span: (span.start, span.end),
                        });
                    }
                })
            }
            Expression::String(t) => {
                let bytes = string_bytes(t).map_err(|message| Error::Syntax {
                    message,
                    span: (span.start, span.end),
                })?;
                Ok(Val {
                    node: string_literal(bytes, span),
                    ty: Ty::Str,
                })
            }
            Expression::Symbol(t) => match t.token().to_string().trim() {
                "true" => Ok(Val {
                    node: bool_lit(true, span),
                    ty: Ty::Bool,
                }),
                "false" => Ok(Val {
                    node: bool_lit(false, span),
                    ty: Ty::Bool,
                }),
                "nil" => Ok(self.nil_val(span)),
                "..." => {
                    let Some(varargs) = self.varargs else {
                        return unsupported("`...` outside a vararg function", span);
                    };
                    Ok(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    })
                }
                other => unsupported(format!("the symbol `{other}`"), span),
            },
            Expression::Parentheses { expression, .. } => {
                // Parentheses truncate to one value.
                self.expr(expression)
            }
            Expression::Function(f) => {
                let id = self.scopes().function_of(f.body());
                self.lower_function(id, f.body(), false)?;
                let record = self.function_value(id, span);
                Ok(Val {
                    node: record.node,
                    ty: Ty::Func(id),
                })
            }
            Expression::TableConstructor(t) => self.table_constructor(t, span),
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                let multi = self.suffixed(c.prefix(), &suffixes, span)?;
                Ok(self.first_of(multi, span))
            }
            Expression::Var(v) => match v {
                Var::Name(token) => self.read_name(token),
                Var::Expression(v) => {
                    let suffixes: Vec<&Suffix> = v.suffixes().collect();
                    let multi = self.suffixed(v.prefix(), &suffixes, span)?;
                    Ok(self.first_of(multi, span))
                }
                _ => unsupported("this variable form", span),
            },
            Expression::UnaryOperator { unop, expression } => {
                let v = self.expr(expression)?;
                let desc = self.describe(expression);
                self.unary_op(unop, v, desc, span)
            }
            Expression::BinaryOperator { lhs, binop, rhs } => self.binary_op(binop, lhs, rhs, span),
            _ => unsupported("this expression form", span),
        }
    }

    /// The single value of what may be several.
    pub(super) fn first_of(&mut self, multi: Multi, span: Span) -> Val {
        match multi {
            Multi::Fixed(mut vals) => {
                if vals.is_empty() {
                    self.nil_val(span)
                } else if vals.len() == 1 {
                    vals.pop().unwrap()
                } else {
                    // Several values of one call: the first carries the
                    // call, the rest are reads of its result.
                    vals.swap_remove(0)
                }
            }
            Multi::Dynamic(node) => Val {
                node: call("zl_first", vec![node], Type::Any, span),
                ty: Ty::Any,
            },
            Multi::None(node) => Val {
                node: block_value(vec![expr_stmt(node)], nil(span), span),
                ty: Ty::Nil,
            },
        }
    }

    /// An expression in last position, where it may supply several
    /// values.
    pub(super) fn expr_multi(&mut self, e: &Expression) -> Result<Multi> {
        let span = span_of(e);
        match e {
            Expression::FunctionCall(c) => {
                let suffixes: Vec<&Suffix> = c.suffixes().collect();
                self.suffixed(c.prefix(), &suffixes, span)
            }
            Expression::Var(Var::Expression(v))
                if matches!(v.suffixes().last(), Some(Suffix::Call(_))) =>
            {
                let suffixes: Vec<&Suffix> = v.suffixes().collect();
                self.suffixed(v.prefix(), &suffixes, span)
            }
            Expression::Symbol(t) if t.token().to_string().trim() == "..." => {
                let Some(varargs) = self.varargs else {
                    return unsupported("`...` outside a vararg function", span);
                };
                Ok(Multi::Dynamic(call(
                    "zl_values_from",
                    vec![var(varargs, self.m.anys(), span), int_lit(1, span)],
                    Type::Any,
                    span,
                )))
            }
            _ => Ok(Multi::Fixed(vec![self.expr(e)?])),
        }
    }

    /// A list of expressions as the values it supplies, the last one
    /// expanded: statements to run first, the fixed values, and the
    /// dynamic tail if any. A last call supplying no values still runs,
    /// after the values before it.
    pub(super) fn expr_list(
        &mut self,
        exprs: &[&Expression],
    ) -> Result<(Vec<St>, Vec<Val>, Option<Node>)> {
        let mut pre = Vec::new();
        let mut vals: Vec<Val> = Vec::new();
        let mut tail = None;
        for (i, e) in exprs.iter().enumerate() {
            if i + 1 == exprs.len() {
                match self.expr_multi(e)? {
                    Multi::Fixed(more) => vals.extend(more),
                    Multi::Dynamic(node) => tail = Some(node),
                    Multi::None(node) => {
                        let held = std::mem::take(&mut vals);
                        for v in held {
                            let h = self.hold(v, &mut pre);
                            vals.push(h);
                        }
                        pre.push(expr_stmt(node));
                    }
                }
            } else {
                vals.push(self.expr(e)?);
            }
        }
        Ok((pre, vals, tail))
    }

    /// The values of an expression list as a `List<Any>`.
    pub(super) fn packed_list(&mut self, exprs: &[&Expression], span: Span) -> Result<Node> {
        let (mut pre, vals, tail) = self.expr_list(exprs)?;
        let mut items = Vec::with_capacity(vals.len());
        for v in vals {
            items.push(self.boxed(v));
        }
        let list = self.array_of(items, &mut pre, span);
        Ok(match tail {
            None => block_value(pre, list, span),
            Some(tail) => {
                let name = self.temp();
                pre.push(let_(name, self.m.anys(), list, span));
                pre.push(expr_stmt(call(
                    "zl_append_values",
                    vec![var(name, self.m.anys(), span), tail],
                    prim(PrimitiveType::Unit),
                    span,
                )));
                block_value(pre, var(name, self.m.anys(), span), span)
            }
        })
    }

    /// `n` values from an expression list, Lua's way: the last
    /// expression's several values fill the rest, missing ones are nil.
    pub(super) fn adjusted(
        &mut self,
        exprs: &[&Expression],
        n: usize,
        span: Span,
    ) -> Result<(Vec<St>, Vec<Val>)> {
        let listed = self.expr_list(exprs)?;
        Ok(self.adjust(listed, n, span))
    }

    /// An expression list's values, as [`Self::expr_list`] gives them,
    /// adjusted to `n`.
    pub(super) fn adjust(
        &mut self,
        listed: (Vec<St>, Vec<Val>, Option<Node>),
        n: usize,
        span: Span,
    ) -> (Vec<St>, Vec<Val>) {
        let (mut pre, vals, tail) = listed;
        let mut out: Vec<Val> = Vec::with_capacity(n);
        for v in vals {
            if out.len() < n {
                let held = self.hold(v, &mut pre);
                out.push(held);
            } else {
                // Extra values run for their effects.
                pre.push(expr_stmt(v.node));
            }
        }
        if let Some(tail) = tail {
            if out.len() < n {
                let name = self.temp();
                pre.push(let_(
                    name,
                    self.m.anys(),
                    call("zl_values", vec![tail], self.m.anys(), span),
                    span,
                ));
                let mut k = 1;
                while out.len() < n {
                    out.push(Val {
                        node: call(
                            "zl_value_at",
                            vec![var(name, self.m.anys(), span), int_lit(k, span)],
                            Type::Any,
                            span,
                        ),
                        ty: Ty::Any,
                    });
                    k += 1;
                }
            } else {
                pre.push(expr_stmt(tail));
            }
        }
        while out.len() < n {
            out.push(self.nil_val(span));
        }
        (pre, out)
    }
}
