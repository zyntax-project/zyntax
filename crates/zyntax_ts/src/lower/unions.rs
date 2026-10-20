use super::*;

pub(super) fn primitive_type(mask: u8) -> Type {
    let mut members = Vec::with_capacity(mask.count_ones() as usize);
    for kind in [ScalarFlow::Number, ScalarFlow::Boolean, ScalarFlow::String] {
        if mask & kind.mask() != 0 {
            members.push(kind.ty());
        }
    }
    Type::Union(members)
}

pub(super) fn primitive_join(a: &Type, b: &Type) -> Option<Type> {
    let mask = ScalarFlow::of(a)?.mask() | ScalarFlow::of(b)?.mask();
    if mask & ScalarFlow::Undefined.mask() != 0 {
        return None;
    }
    Some(primitive_type(mask))
}

pub(super) fn accepts(actual: &Type, expected: &Type) -> bool {
    if let Type::Union(members) = expected {
        match actual {
            Type::Union(actual) => actual.iter().all(|a| members.contains(a)),
            _ => members.contains(actual),
        }
    } else {
        false
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    pub(super) fn union_comparison(
        &mut self,
        left: Expr,
        right: Expr,
        negate: bool,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        if let (Some(a), Some(b)) = (ScalarFlow::of(&left.ty), ScalarFlow::of(&right.ty)) {
            if a.mask() & b.mask() == 0 {
                return self.error(ErrorKind::Type, "comparison has no overlapping types", at);
            }
        }
        let (left_save, left) = self.capture_reference_value(Box::new(left), at);
        let (right_save, right) = self.capture_reference_value(Box::new(right), at);
        let mut value = self.strict_equal(left, right, at)?;
        if negate {
            value = node(
                TypedExpression::Unary(TypedUnary {
                    op: UnaryOp::Not,
                    operand: Box::new(value),
                }),
                boolean(),
                at,
            );
        }
        Ok(node(
            TypedExpression::Block(TypedBlock {
                statements: vec![left_save, right_save, expression(value)],
                span: span(at),
            }),
            boolean(),
            at,
        ))
    }
    fn strict_equal(&mut self, left: Expr, right: Expr, at: oxc_span::Span) -> Result<Expr> {
        if matches!(left.ty, Type::Union(_)) {
            return self.union_match(left, u8::MAX, boolean(), at, |this, payload| {
                this.strict_equal(payload, right.clone(), at)
            });
        }
        if matches!(right.ty, Type::Union(_)) {
            return self.strict_equal(right, left, at);
        }
        if left.ty != right.ty {
            return Ok(node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                boolean(),
                at,
            ));
        }
        if left.ty == string() {
            return Ok(self.call("$String$equals", vec![left, right], boolean(), at));
        }
        Ok(binary(BinaryOp::Eq, left, right, boolean(), at))
    }

    pub(super) fn check_union_source(
        &self,
        source: &js::Expression<'ast>,
        actual: &Type,
        expected: &Type,
    ) -> Result<()> {
        if matches!(expected, Type::Union(_))
            && *actual == string()
            && !self.immortal_string(source)
        {
            return self.unsupported(
                "union string payload without a proven static lifetime",
                source.span(),
            );
        }
        Ok(())
    }

    pub(super) fn union_match(
        &mut self,
        value: Expr,
        selected: u8,
        result: Type,
        at: oxc_span::Span,
        mut body: impl FnMut(&mut Self, Expr) -> Result<Expr>,
    ) -> Result<Expr> {
        let Type::Union(members) = &value.ty else {
            unreachable!()
        };
        let mut arms = Vec::with_capacity(members.len());
        for (index, ty) in members.iter().enumerate() {
            if ScalarFlow::of(ty).unwrap().mask() & selected == 0 {
                continue;
            }
            let name = self.fresh("union_payload");
            let expression = body(self, var(name, ty.clone(), at))?;
            arms.push(TypedMatchArm {
                pattern: Box::new(node(
                    TypedPattern::Enum {
                        name: self.name("Union"),
                        variant: zyntax_compiler::hir::union_variant_name(index),
                        fields: vec![node(
                            TypedPattern::Identifier {
                                name,
                                mutability: Mutability::Immutable,
                            },
                            ty.clone(),
                            at,
                        )],
                    },
                    value.ty.clone(),
                    at,
                )),
                guard: None,
                body: Box::new(expression),
            });
        }
        Ok(node(
            TypedExpression::Match(TypedMatchExpr {
                scrutinee: Box::new(value),
                arms,
            }),
            result,
            at,
        ))
    }

    pub(super) fn narrow_type_name(
        &mut self,
        id: &js::IdentifierReference<'ast>,
        text: &str,
        equality: bool,
    ) {
        let selected = match text {
            "number" => ScalarFlow::Number,
            "boolean" => ScalarFlow::Boolean,
            "string" => ScalarFlow::String,
            "undefined" => ScalarFlow::Undefined,
            _ => return,
        };
        let Ok(binding) = self.binding(id.name.as_str(), id.span) else {
            return;
        };
        let possible = match &binding.ty {
            Type::Optional(inner) => {
                ScalarFlow::of(inner).map(|v| v.mask() | ScalarFlow::Undefined.mask())
            }
            Type::Union(_) => ScalarFlow::of(&binding.ty).map(ScalarFlow::mask),
            _ => None,
        };
        let Some(possible) = possible else { return };
        let name = binding.name;
        let possible = self.flow.get(&name).map_or(possible, |v| v.mask());
        let mask = possible
            & if equality {
                selected.mask()
            } else {
                !selected.mask()
            };
        if mask == possible {
            return;
        }
        if let Some(flow) = ScalarFlow::from_mask(mask) {
            self.flow.insert(name, flow);
        }
    }
}
