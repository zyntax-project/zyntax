//! Top-level `const` declarations with compile-time values are substituted where used.

use super::*;
use std::borrow::Cow;

/// Syntactic literal types only; evaluating an expression does not make its type literal.
pub(super) fn literal_text<'a>(source: &'a js::Expression<'_>) -> Option<Cow<'a, str>> {
    match source {
        js::Expression::StringLiteral(s) if !s.lone_surrogates => {
            Some(Cow::Borrowed(s.value.as_str()))
        }
        js::Expression::ParenthesizedExpression(p) => literal_text(&p.expression),
        js::Expression::TemplateLiteral(t) => {
            let mut text = String::new();
            for (index, quasi) in t.quasis.iter().enumerate() {
                if quasi.lone_surrogates {
                    return None;
                }
                text.push_str(quasi.value.cooked.as_ref()?.as_str());
                if let Some(expression) = t.expressions.get(index) {
                    text.push_str(&literal_interpolation(expression)?);
                }
            }
            Some(Cow::Owned(text))
        }
        _ => None,
    }
}

fn literal_interpolation<'a>(source: &'a js::Expression<'_>) -> Option<Cow<'a, str>> {
    let number = match source {
        js::Expression::NumericLiteral(n) => Some(n.value),
        js::Expression::UnaryExpression(u) if matches!(u.operator.as_str(), "+" | "-") => {
            let js::Expression::NumericLiteral(n) = &u.argument else {
                return None;
            };
            Some(if u.operator.as_str() == "-" {
                -n.value
            } else {
                n.value
            })
        }
        js::Expression::ParenthesizedExpression(p) => return literal_interpolation(&p.expression),
        _ => return literal_text(source),
    }?;
    if !number.is_finite() {
        return None;
    }
    Some(Cow::Owned(if number == 0.0 {
        "0".to_owned()
    } else {
        dragonbox_ecma::Buffer::new().format(number).to_owned()
    }))
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Constant {
    Number(f64),
    Bool(bool),
    Text(InternedString),
}

impl Constant {
    fn ty(self) -> Type {
        match self {
            Self::Number(_) => number(),
            Self::Bool(_) => boolean(),
            Self::Text(_) => string(),
        }
    }
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// Records the top-level constants whose initializers fold to a literal.
    pub(super) fn collect_constants(&mut self, program: &'program js::Program<'ast>) {
        for item in &program.body {
            let js::Statement::VariableDeclaration(declaration) = item else {
                continue;
            };
            if declaration.kind.as_str() != "const" {
                continue;
            }
            for d in &declaration.declarations {
                let (js::BindingPattern::BindingIdentifier(id), Some(init)) = (&d.id, &d.init)
                else {
                    continue;
                };
                let Some(value) = self.constant_value(init) else {
                    continue;
                };
                if let Some(annotation) = &d.type_annotation {
                    if !self
                        .ty(&annotation.type_annotation)
                        .is_ok_and(|(ty, _)| ty == value.ty())
                    {
                        continue;
                    }
                }
                let name = self.name(id.name.as_str());
                self.constants.insert(name, value);
            }
        }
    }

    fn constant_value(&self, e: &js::Expression<'ast>) -> Option<Constant> {
        use Constant::{Bool, Number, Text};
        match e {
            js::Expression::NumericLiteral(n) => Some(Number(n.value)),
            js::Expression::BooleanLiteral(b) => Some(Bool(b.value)),
            js::Expression::StringLiteral(s) if !s.lone_surrogates => {
                Some(Text(self.name(s.value.as_str())))
            }
            js::Expression::ParenthesizedExpression(p) => self.constant_value(&p.expression),
            js::Expression::Identifier(i) => {
                self.constants.get(&self.name(i.name.as_str())).copied()
            }
            js::Expression::StaticMemberExpression(m) if !m.optional => {
                self.math_constant_value(m).map(Number)
            }
            js::Expression::UnaryExpression(u) => {
                match (u.operator.as_str(), self.constant_value(&u.argument)?) {
                    ("-", Number(n)) => Some(Number(-n)),
                    ("+", Number(n)) => Some(Number(n)),
                    ("!", Bool(b)) => Some(Bool(!b)),
                    _ => None,
                }
            }
            js::Expression::BinaryExpression(b) => {
                match (
                    self.constant_value(&b.left)?,
                    self.constant_value(&b.right)?,
                ) {
                    (Number(x), Number(y)) => Some(match b.operator.as_str() {
                        "+" => Number(x + y),
                        "-" => Number(x - y),
                        "*" => Number(x * y),
                        "/" => Number(x / y),
                        "%" => Number(x % y),
                        "<" => Bool(x < y),
                        "<=" => Bool(x <= y),
                        ">" => Bool(x > y),
                        ">=" => Bool(x >= y),
                        "===" | "==" => Bool(x == y),
                        "!==" | "!=" => Bool(x != y),
                        _ => return None,
                    }),
                    (Text(x), Text(y)) if b.operator.as_str() == "+" => {
                        let mut joined = x.resolve_global()?;
                        joined.push_str(&y.resolve_global()?);
                        Some(Text(intern(&joined)))
                    }
                    _ => None,
                }
            }
            js::Expression::LogicalExpression(l) => {
                match (
                    self.constant_value(&l.left)?,
                    self.constant_value(&l.right)?,
                ) {
                    (Bool(x), Bool(y)) => match l.operator.as_str() {
                        "&&" => Some(Bool(x && y)),
                        "||" => Some(Bool(x || y)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The substituted literal for a name that no local binding shadows.
    pub(super) fn constant_expr(&self, name: &'ast str, at: oxc_span::Span) -> Option<Expr> {
        let value = *self.constants.get(&self.name(name))?;
        if self.scopes.iter().any(|s| s.contains_key(&self.name(name))) {
            return None;
        }
        Some(match value {
            Constant::Number(n) => num(n, at),
            Constant::Bool(b) => node(
                TypedExpression::Literal(TypedLiteral::Bool(b)),
                boolean(),
                at,
            ),
            Constant::Text(text) => node(
                TypedExpression::Literal(TypedLiteral::String(text)),
                string(),
                at,
            ),
        })
    }

    pub(super) fn constant_integral(&self, name: &'ast str) -> Option<Integral> {
        let Some(Constant::Number(n)) = self.constants.get(&self.name(name)).copied() else {
            return None;
        };
        let local = self.scopes.iter().any(|s| s.contains_key(&self.name(name)));
        (!local && n >= 0.0 && n <= f64::from(u32::MAX) && n.fract() == 0.0)
            .then_some(Integral::Constant(n as u64))
    }

    pub(super) fn constant_text(&self, name: &'ast str) -> bool {
        matches!(
            self.constants.get(&self.name(name)),
            Some(Constant::Text(_))
        ) && !self.scopes.iter().any(|s| s.contains_key(&self.name(name)))
    }
}
