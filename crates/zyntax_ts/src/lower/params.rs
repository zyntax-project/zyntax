//! Optional, default and rest parameters. A parameter a caller may omit is a native optional
//! slot that the caller fills with `undefined`; the callee resolves a default on entry, after
//! the parameters before it. A rest parameter is an array the caller builds from its trailing
//! arguments.

use super::*;

/// How a signature's callers may supply fewer or more arguments than it has parameters.
#[derive(Clone, Copy, Default)]
pub(super) struct Arity {
    /// Arguments every call supplies; later fixed parameters may be omitted.
    pub required: usize,
    /// The last parameter collects the remaining arguments into an array.
    pub rest: bool,
}

impl<'ast, 'program> Lowerer<'ast, 'program> {
    /// The native type of a parameter written `name?: T` or `name: T = value`: the body of an
    /// optional parameter sees `T | undefined`, and a default replaces `undefined` on entry.
    pub(super) fn omittable_parameter(
        &self,
        p: &js::FormalParameter<'ast>,
    ) -> Result<(Type, Option<InternedString>)> {
        let (ty, view) = match (&p.type_annotation, &p.initializer) {
            (Some(annotation), _) => self.ty(&annotation.type_annotation)?,
            (None, Some(init)) => match self.default_type(init) {
                Some(ty) => (ty, None),
                None => {
                    return self.unsupported("inferring a default parameter's type", p.span);
                }
            },
            (None, None) => {
                return self.error(ErrorKind::Type, "parameter implicitly has type any", p.span);
            }
        };
        let inner = match &ty {
            Type::Optional(inner) => inner.as_ref().clone(),
            other => other.clone(),
        };
        if !is_scalar(&inner) || view.is_some() {
            return self.unsupported("optional or default parameters of this type", p.span);
        }
        Ok((Type::Optional(Box::new(inner)), None))
    }

    /// A default's type when the parameter has no annotation, widened as TypeScript widens it.
    fn default_type(&self, init: &js::Expression<'ast>) -> Option<Type> {
        match init {
            js::Expression::NumericLiteral(_) => Some(number()),
            js::Expression::BooleanLiteral(_) => Some(boolean()),
            js::Expression::UnaryExpression(u) if matches!(u.operator.as_str(), "-" | "+") => {
                matches!(u.argument, js::Expression::NumericLiteral(_)).then(number)
            }
            js::Expression::ParenthesizedExpression(p) => self.default_type(&p.expression),
            _ => None,
        }
    }

    /// The rest parameter's array type.
    pub(super) fn rest_parameter(
        &self,
        rest: &js::FormalParameterRest<'ast>,
    ) -> Result<(InternedString, Type)> {
        let js::BindingPattern::BindingIdentifier(id) = &rest.rest.argument else {
            return self.unsupported("destructuring rest parameters", rest.span);
        };
        let Some(annotation) = &rest.type_annotation else {
            return self.error(
                ErrorKind::Type,
                "rest parameter implicitly has type any[]",
                rest.span,
            );
        };
        let (ty, _) = self.ty(&annotation.type_annotation)?;
        if element_type(&ty).is_none() {
            return self.error(
                ErrorKind::Type,
                "a rest parameter must be of an array type",
                rest.span,
            );
        }
        Ok((self.name(id.name.as_str()), ty))
    }

    /// The arguments past the fixed parameters, as the array a rest parameter receives.
    pub(super) fn rest_arguments(
        &mut self,
        arguments: &[js::Argument<'ast>],
        ty: &Type,
        at: oxc_span::Span,
    ) -> Result<Expr> {
        let element = element_type(ty).unwrap().clone();
        let mut values = Vec::with_capacity(arguments.len());
        for argument in arguments {
            let Some(source) = argument.as_expression() else {
                return self.unsupported("spread arguments", at);
            };
            let value = self.expr_expected(source, &element)?;
            self.expect(self.source_type(source, &value.ty), &element, source.span())?;
            if element == string() && !self.immortal_string(source) {
                return self.unsupported(
                    "array storage of strings without a proven static lifetime",
                    source.span(),
                );
            }
            values.push(self.adapt_value(value, &element, source.span())?);
        }
        Ok(node(TypedExpression::Array(values), ty.clone(), at))
    }
}
