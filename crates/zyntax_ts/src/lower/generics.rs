use super::*;

pub(super) fn unresolved(ty: &Type) -> bool {
    match ty {
        Type::TypeVar(_) => true,
        Type::Array { element_type, .. } | Type::Optional(element_type) => unresolved(element_type),
        _ => false,
    }
}
impl<'ast, 'program> Lowerer<'ast, 'program> {
    fn active_rules(&self) -> &[CheckerTypeParameter] {
        self.active_type_rules.as_deref().unwrap_or(&[])
    }
    pub(super) fn constrained_scalar(&self, value: Expr, at: oxc_span::Span) -> Expr {
        let ty = self.constraint_type(&value.ty);
        if *ty != value.ty
            && matches!(
                ty,
                Type::Primitive(PrimitiveType::F64 | PrimitiveType::Bool | PrimitiveType::String)
            )
        {
            let ty = ty.clone();
            return node(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(value),
                    target_type: ty.clone(),
                }),
                ty.clone(),
                at,
            );
        }
        value
    }
    pub(super) fn type_argument_view(
        &self,
        sig: &Signature,
        call: &js::CallExpression<'ast>,
        mut index: usize,
    ) -> Option<InternedString> {
        for _ in 0..sig.type_params.len() {
            if let Some(explicit) = &call.type_arguments {
                if let Some(ty) = explicit.params.get(index) {
                    return self.ty(ty).ok()?.1;
                }
            } else if let Some(argument) = sig
                .params
                .iter()
                .position(|p| p.ty == sig.type_params[index].1)
            {
                return call
                    .arguments
                    .get(argument)
                    .and_then(|a| a.as_expression())
                    .and_then(|a| self.view_of(a));
            }
            let rule = sig.type_rules().get(index)?;
            let selected = rule.default.as_ref().or(rule.constraint.as_ref())?;
            if selected.view.is_some() {
                return selected.view;
            }
            index = sig
                .type_params
                .iter()
                .position(|(_, ty)| *ty == selected.ty)?;
        }
        None
    }

    pub(super) fn constraint_type<'a>(&'a self, mut ty: &'a Type) -> &'a Type {
        for _ in 0..self.active_types.len() {
            let Some(index) = self
                .active_types
                .iter()
                .position(|(_, parameter)| parameter == ty)
            else {
                break;
            };
            let Some(bound) = self
                .active_rules()
                .get(index)
                .and_then(|r| r.constraint.as_ref())
            else {
                break;
            };
            ty = &bound.ty;
        }
        ty
    }

    fn constraint_view<'a>(
        &'a self,
        mut ty: &'a Type,
        mut view: Option<InternedString>,
    ) -> Option<InternedString> {
        for _ in 0..self.active_types.len() {
            if view.is_some() {
                break;
            }
            let Some(index) = self
                .active_types
                .iter()
                .position(|(_, parameter)| parameter == ty)
            else {
                break;
            };
            let Some(bound) = self
                .active_rules()
                .get(index)
                .and_then(|r| r.constraint.as_ref())
            else {
                break;
            };
            ty = &bound.ty;
            view = bound.view;
        }
        view
    }

    pub(super) fn type_parameter_rules(
        &mut self,
        parameters: &js::TSTypeParameterDeclaration<'ast>,
    ) -> Result<Option<Arc<[CheckerTypeParameter]>>> {
        if parameters
            .params
            .iter()
            .all(|p| p.constraint.is_none() && p.default.is_none())
        {
            return Ok(None);
        }
        let mut rules = Vec::with_capacity(parameters.params.len());
        let mut optional = false;
        for (index, p) in parameters.params.iter().enumerate() {
            if optional && p.default.is_none() {
                return self.error(
                    ErrorKind::Type,
                    "required type parameter follows an optional type parameter",
                    p.span,
                );
            }
            optional |= p.default.is_some();
            let convert = |source: &js::TSType<'ast>| self.type_parameter_annotation(source);
            let constraint = p.constraint.as_ref().map(convert).transpose()?;
            let default = p.default.as_ref().map(convert).transpose()?;
            if let Some(default) = &default {
                if self.active_types[index..]
                    .iter()
                    .any(|(_, parameter)| contains(&default.ty, parameter))
                {
                    return self.error(
                        ErrorKind::Type,
                        "type parameter default references itself or a later parameter",
                        p.span,
                    );
                }
            }
            rules.push(CheckerTypeParameter {
                constraint,
                default,
            });
        }
        self.active_type_rules = Some(rules.into());
        for (index, p) in parameters.params.iter().enumerate() {
            let mut current = index;
            for step in 0..=parameters.params.len() {
                let Some(bound) = self.active_rules()[current].constraint.as_ref() else {
                    break;
                };
                let Some(next) = self.active_types.iter().position(|(_, ty)| *ty == bound.ty)
                else {
                    break;
                };
                if step == parameters.params.len() {
                    return self.error(
                        ErrorKind::Type,
                        "type parameter has a circular constraint",
                        p.span,
                    );
                }
                current = next;
            }
            let rule = &self.active_rules()[index];
            if let (Some(default), Some(bound)) = (&rule.default, &rule.constraint) {
                self.expect(&default.ty, &bound.ty, p.span)?;
                self.check_view(
                    self.constraint_view(&default.ty, default.view),
                    self.constraint_view(&bound.ty, bound.view),
                    p.span,
                )?;
            }
        }
        Ok(self.active_type_rules.clone())
    }

    fn type_parameter_annotation(&self, source: &js::TSType<'ast>) -> Result<CheckerType> {
        match self.ty(source) {
            Err(error) if error.kind == ErrorKind::Type => {
                if let js::TSType::TSTypeReference(reference) = source
                    && let js::TSTypeName::IdentifierReference(name) = &reference.type_name
                    && matches!(
                        name.name.as_str(),
                        "Date" | "Number" | "String" | "Boolean" | "Object" | "RegExp" | "Function"
                    )
                {
                    return self.unsupported(
                        &format!("generic rules involving library type `{}`", name.name),
                        source.span(),
                    );
                }
                Err(error)
            }
            Ok((ty, view)) => {
                self.plain_enum(view, "enum types in generic rules", source.span())?;
                Ok(CheckerType { ty, view })
            }
            Err(error) => Err(error),
        }
    }

    pub(super) fn complete_type_arguments(
        &self,
        sig: &Signature,
        types: &mut [Option<(Type, Option<InternedString>)>],
        at: oxc_span::Span,
    ) -> Result<()> {
        for (index, rule) in sig.type_rules().iter().enumerate() {
            if types[index].is_none() {
                if let Some(default) = &rule.default {
                    types[index] = Some(self.instantiate_type(
                        &default.ty,
                        default.view,
                        &sig.type_params,
                        types,
                        at,
                    )?);
                }
            }
        }
        for _ in 0..types.len() {
            let mut changed = false;
            for (index, rule) in sig.type_rules().iter().enumerate() {
                if types[index].is_none() {
                    if let Some(bound) = &rule.constraint {
                        if let Ok(value) = self.instantiate_type(
                            &bound.ty,
                            bound.view,
                            &sig.type_params,
                            types,
                            at,
                        ) {
                            types[index] = Some(value);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Ok(())
    }

    pub(super) fn check_type_arguments(
        &self,
        sig: &Signature,
        types: &[Option<(Type, Option<InternedString>)>],
        at: oxc_span::Span,
    ) -> Result<()> {
        for (index, rule) in sig.type_rules().iter().enumerate() {
            if let Some(bound) = &rule.constraint {
                let (expected, view) =
                    self.instantiate_type(&bound.ty, bound.view, &sig.type_params, types, at)?;
                let Some((actual, actual_view)) = &types[index] else {
                    return self.unsupported("unresolved generic type argument", at);
                };
                self.expect(actual, &expected, at)?;
                self.check_view(
                    self.constraint_view(actual, *actual_view),
                    self.constraint_view(&expected, view),
                    at,
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn infer_arguments(
        &self,
        expected: &Type,
        actual: &Type,
        view: Option<InternedString>,
        parameters: &[(InternedString, Type)],
        types: &mut [Option<(Type, Option<InternedString>)>],
    ) {
        if let Some(index) = parameters
            .iter()
            .position(|(_, parameter)| parameter == expected)
        {
            if types[index].is_none() {
                types[index] = Some((actual.clone(), view));
            }
        } else if let (
            Type::Array {
                element_type: expected,
                ..
            },
            Type::Array {
                element_type: actual,
                ..
            },
        ) = (expected, actual)
        {
            self.infer_arguments(expected, actual, None, parameters, types);
        } else if let (Type::Optional(expected), Type::Optional(actual)) = (expected, actual) {
            self.infer_arguments(expected, actual, None, parameters, types);
        }
    }
    pub(super) fn instantiate_type(
        &self,
        ty: &Type,
        view: Option<InternedString>,
        parameters: &[(InternedString, Type)],
        types: &[Option<(Type, Option<InternedString>)>],
        at: oxc_span::Span,
    ) -> Result<(Type, Option<InternedString>)> {
        if let Some(index) = parameters.iter().position(|(_, parameter)| parameter == ty) {
            return types[index].clone().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unsupported,
                    "unresolved generic type argument",
                    at,
                )
            });
        }
        let ty = match ty {
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(
                    self.instantiate_type(element_type, None, parameters, types, at)?
                        .0,
                ),
                size: size.clone(),
                nullability: *nullability,
            },
            Type::Optional(inner) => Type::Optional(Box::new(
                self.instantiate_type(inner, None, parameters, types, at)?.0,
            )),
            _ => ty.clone(),
        };
        Ok((ty, view))
    }
}

fn contains(ty: &Type, parameter: &Type) -> bool {
    if ty == parameter {
        return true;
    }
    match ty {
        Type::Array { element_type, .. } | Type::Optional(element_type) => {
            contains(element_type, parameter)
        }
        Type::Union(members) | Type::Tuple(members) => {
            members.iter().any(|ty| contains(ty, parameter))
        }
        Type::Function {
            params,
            return_type,
            ..
        } => params.iter().any(|p| contains(&p.ty, parameter)) || contains(return_type, parameter),
        _ => false,
    }
}
