//! Builtin args for Lua lowering.

use super::*;

impl<'m, 'a> Lowerer<'m, 'a> {
    /// A `math` call whose arguments' types decide its result, as the
    /// typer decided it too: the arithmetic inline, nothing boxed on
    /// the way in or out. `None` leaves the call to the library.
    pub(super) fn math_call(
        &mut self,
        b: &Builtin,
        vals: &mut Vec<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Option<Multi> {
        if b.lib != "math" {
            return None;
        }
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let bool_t = prim(PrimitiveType::Bool);
        let tys: Vec<Ty> = vals.iter().map(|v| v.ty).collect();
        let ty = match types::math_result(b, &tys) {
            Some(t @ (Ty::Int | Ty::Float | Ty::Number)) => t,
            _ => return None,
        };
        let ir = self.ir(ty);
        let mut args = std::mem::take(vals).into_iter();
        let mut arg = || args.next().expect("an argument");
        let node = match (b.name, ty) {
            ("abs", Ty::Int) => {
                let x = arg().node;
                let negative = binary(BinaryOp::Lt, x.clone(), int_lit(0, span), bool_t, span);
                let negated = binary(BinaryOp::Sub, int_lit(0, span), x.clone(), i64_t, span);
                if_value(negative, negated, x, ir, span)
            }
            ("abs", Ty::Float) => call("abs", vec![arg().node], f64_t, span),
            // Each part's own absolute value: the part the kind does
            // not use stays zero.
            ("abs", Ty::Number) => {
                let (mut held, n) = self.number_parts(arg());
                pre.append(&mut held);
                let negative = binary(BinaryOp::Lt, n.int.clone(), int_lit(0, span), bool_t, span);
                let negated = binary(
                    BinaryOp::Sub,
                    int_lit(0, span),
                    n.int.clone(),
                    i64_t.clone(),
                    span,
                );
                let int = if_value(negative, negated, n.int, i64_t, span);
                let float = call("abs", vec![n.float], f64_t, span);
                number_value(n.tag, int, float, span)
            }
            ("floor" | "ceil", Ty::Int) => arg().node,
            // A float rounds to the integer it makes when that fits,
            // else stays a float; an integer is itself.
            ("floor" | "ceil", Ty::Number) => {
                let x = arg();
                let round = if b.name == "floor" {
                    "floor"
                } else {
                    "zl_ceil_f64"
                };
                if x.ty == Ty::Float {
                    let r = call(round, vec![x.node], f64_t, span);
                    self.number_of_integral(r, pre, span)
                } else {
                    let (mut held, n) = self.number_parts(x);
                    pre.append(&mut held);
                    let r = call(round, vec![n.float.clone()], f64_t, span);
                    let mut inner = Vec::new();
                    let rounded = self.number_of_integral(r, &mut inner, span);
                    let rounded = block_value(inner, rounded, span);
                    if_value(
                        n.is_int(span),
                        number_value(n.tag.clone(), n.int, n.float, span),
                        rounded,
                        ir,
                        span,
                    )
                }
            }
            ("max" | "min", _) => {
                // The best so far gives way when it is less than the
                // next (`max`), or the next is less (`min`): compared
                // with a NaN neither is, so the first stays.
                let mut best = arg();
                if ty == Ty::Number {
                    let node = self.coerce(best, ty);
                    best = self.hold(Val { node, ty }, pre);
                }
                for next in args {
                    let next = if ty == Ty::Number {
                        let node = self.coerce(next, ty);
                        self.hold(Val { node, ty }, pre)
                    } else {
                        next
                    };
                    let (l, r) = if b.name == "max" {
                        (best.clone(), next.clone())
                    } else {
                        (next.clone(), best.clone())
                    };
                    let moves = if ty == Ty::Number {
                        self.number_compare(l, r, Compare::Lt, span)
                    } else {
                        binary(BinaryOp::Lt, l.node, r.node, bool_t.clone(), span)
                    };
                    let picked = Val {
                        node: if_value(moves, next.node, best.node, ir.clone(), span),
                        ty,
                    };
                    best = self.hold(picked, pre);
                }
                best.node
            }
            ("fmod", Ty::Int) => {
                let (a, b) = (arg().node, arg().node);
                call("zl_fmod_i64", vec![a, b], i64_t, span)
            }
            ("fmod", Ty::Float) => {
                let (a, b) = (arg(), arg());
                let a = self.coerce(a, Ty::Float);
                let b = self.coerce(b, Ty::Float);
                binary(BinaryOp::Rem, a, b, f64_t, span)
            }
            _ => unreachable!("math_result decided the shape"),
        };
        let v = Val { node, ty };
        let v = if self.call_can_raise(&v.node) {
            self.guard(v)
        } else {
            v
        };
        let node = block_value(std::mem::take(pre), v.node, span);
        Some(Multi::Fixed(vec![Val { node, ty }]))
    }

    /// A literal format with one unadorned conversion can use the
    /// argument type directly instead of entering the variadic formatter.
    pub(super) fn simple_string_format(
        &mut self,
        b: &Builtin,
        vals: &mut Vec<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Option<Multi> {
        if b.lib != "string" || b.name != "format" || vals.len() != 2 {
            return None;
        }
        let format = self.literal_key(&vals[0])?;
        let conversion = format.find('%')?;
        let specifier = format.as_bytes().get(conversion + 1).copied()?;
        if format[conversion + 2..].contains('%') {
            return None;
        }
        let arg = vals.pop().expect("the format argument");
        let accepted = match specifier {
            b'd' => arg.ty == Ty::Int,
            b's' => matches!(arg.ty, Ty::Str | Ty::Int | Ty::Float | Ty::Number),
            _ => false,
        };
        if !accepted {
            vals.push(arg);
            return None;
        }
        vals.clear();
        let string_t = prim(PrimitiveType::String);
        let prefix = str_lit(&format[..conversion], span);
        let suffix = str_lit(&format[conversion + 2..], span);
        let value = binary(
            BinaryOp::Add,
            binary(
                BinaryOp::Add,
                prefix,
                self.text_of(arg),
                string_t.clone(),
                span,
            ),
            suffix,
            string_t,
            span,
        );
        Some(Multi::Fixed(vec![Val {
            node: block_value(std::mem::take(pre), value, span),
            ty: Ty::Str,
        }]))
    }

    /// `print` can render statically known strings and numbers without
    /// packaging a variadic list or consulting dynamic `tostring` rules.
    pub(super) fn typed_print(
        &mut self,
        b: &Builtin,
        is_method: bool,
        vals: &mut Vec<Val>,
        pre: &mut Vec<St>,
        span: Span,
    ) -> Option<Multi> {
        if !b.lib.is_empty()
            || b.name != "print"
            || vals
                .iter()
                .any(|v| !matches!(v.ty, Ty::Str | Ty::Int | Ty::Float | Ty::Number))
        {
            return None;
        }
        let string_t = prim(PrimitiveType::String);
        let mut text = str_lit("", span);
        for (i, value) in std::mem::take(vals).into_iter().enumerate() {
            if i != 0 {
                text = binary(
                    BinaryOp::Add,
                    text,
                    str_lit("\t", span),
                    string_t.clone(),
                    span,
                );
            }
            text = binary(
                BinaryOp::Add,
                text,
                self.text_of(value),
                string_t.clone(),
                span,
            );
        }
        text = binary(BinaryOp::Add, text, str_lit("\n", span), string_t, span);
        let after = self.around_builtin(b, is_method, span, pre);
        pre.push(expr_stmt(call(
            "zl_stdout_write",
            vec![text],
            prim(PrimitiveType::I64),
            span,
        )));
        pre.extend(after);
        Some(Multi::None(block_value(
            std::mem::take(pre),
            nil(span),
            span,
        )))
    }

    /// An integral float as a number: the integer when it fits, else
    /// the float; `pre` takes the float, held.
    pub(super) fn number_of_integral(&mut self, r: Node, pre: &mut Vec<St>, span: Span) -> Node {
        let f64_t = prim(PrimitiveType::F64);
        let i64_t = prim(PrimitiveType::I64);
        let bool_t = prim(PrimitiveType::Bool);
        let r = self
            .hold(
                Val {
                    node: r,
                    ty: Ty::Float,
                },
                pre,
            )
            .node;
        // `-2^63 <= r < 2^63`, false for a NaN.
        let fits = binary(
            BinaryOp::And,
            binary(
                BinaryOp::Ge,
                r.clone(),
                float_lit(-9223372036854775808.0, span),
                bool_t.clone(),
                span,
            ),
            binary(
                BinaryOp::Lt,
                r.clone(),
                float_lit(9223372036854775808.0, span),
                bool_t.clone(),
                span,
            ),
            bool_t,
            span,
        );
        let fits = self
            .hold(
                Val {
                    node: fits,
                    ty: Ty::Bool,
                },
                pre,
            )
            .node;
        let int = if_value(
            fits.clone(),
            cast(r.clone(), i64_t.clone(), span),
            int_lit(0, span),
            i64_t,
            span,
        );
        let float = if_value(fits.clone(), float_lit(0.0, span), r, f64_t, span);
        number_value(tag_of_is_int(fits, span), int, float, span)
    }

    /// One argument of a builtin as its parameter takes it.
    pub(super) fn builtin_arg(
        &mut self,
        p: &Param,
        v: Option<Val>,
        what: Node,
        span: Span,
    ) -> Node {
        let i64_t = prim(PrimitiveType::I64);
        let f64_t = prim(PrimitiveType::F64);
        let str_t = prim(PrimitiveType::String);
        match p {
            Param::Any => match v {
                Some(v) => self.boxed(v),
                None => nil(span),
            },
            Param::Value => match v {
                Some(v) => self.boxed(v),
                None => call("zl_arg_value_missing", vec![what], Type::Any, span),
            },
            Param::Expected(kind) => match v {
                Some(v) => self.boxed(v),
                None => call(
                    "zl_arg_expected_missing",
                    vec![what, str_lit(kind, span)],
                    Type::Any,
                    span,
                ),
            },
            Param::Int => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_int", vec![b, what], i64_t, span)
                }
                None => call("zl_arg_int_missing", vec![what], i64_t, span),
            },
            Param::Float => match v {
                Some(v) if v.ty.is_number() => self.coerce(v, Ty::Float),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_float", vec![b, what], f64_t, span)
                }
                None => call("zl_arg_float_missing", vec![what], f64_t, span),
            },
            Param::Str => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty.is_number() => self.text_of(v),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_str", vec![b, what], str_t, span)
                }
                None => call("zl_arg_str_missing", vec![what], str_t, span),
            },
            // A string by its type is the typed implementation's; any
            // other value, possibly a buffer, is its `$any` form's.
            Param::Bytes => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty.is_number() => self.text_of(v),
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_arg_bytes", vec![b, what], Type::Any, span)
                }
                None => call("zl_arg_str_missing", vec![what], str_t, span),
            },
            Param::Table => match v {
                Some(v) if v.ty == Ty::Table => v.node,
                // A shaped table is one unless it is nil, when the
                // check raises as for any other value.
                Some(v) if matches!(v.ty, Ty::Shape(_)) => {
                    let table_t = self.ir(Ty::Table);
                    let mut pre = Vec::new();
                    let t = self.hold(v, &mut pre);
                    let is_null = binary(
                        BinaryOp::Eq,
                        t.node.clone(),
                        null(table_t.clone(), span),
                        prim(PrimitiveType::Bool),
                        span,
                    );
                    block_value(
                        pre,
                        if_value(
                            is_null,
                            call("zl_as_table", vec![nil(span), what], table_t.clone(), span),
                            t.node,
                            table_t,
                            span,
                        ),
                        span,
                    )
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call("zl_as_table", vec![b, what], self.ir(Ty::Table), span)
                }
                None => call("zl_as_table_missing", vec![what], self.ir(Ty::Table), span),
            },
            Param::OptInt(d) => match v {
                Some(v) if v.ty == Ty::Int => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], int_lit(*d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_int",
                        vec![b, int_lit(*d, span), what],
                        i64_t,
                        span,
                    )
                }
                None => int_lit(*d, span),
            },
            Param::OptStr(d) => match v {
                Some(v) if v.ty == Ty::Str => v.node,
                Some(v) if v.ty == Ty::Nil => {
                    block_value(vec![expr_stmt(v.node)], str_lit(d, span), span)
                }
                Some(v) => {
                    let b = self.boxed(v);
                    call(
                        "zl_arg_opt_str",
                        vec![b, str_lit(d, span), what],
                        str_t,
                        span,
                    )
                }
                None => str_lit(d, span),
            },
            Param::Rest => unreachable!("the rest is gathered by the caller"),
        }
    }
}
