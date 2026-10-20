use super::*;

pub(super) fn unresolved(ty: &Type) -> bool {
    match ty {
        Type::TypeVar(_) => true,
        Type::Array { element_type, .. } | Type::Optional(element_type) => unresolved(element_type),
        _ => false,
    }
}
impl<'ast, 'program> Lowerer<'ast, 'program> {
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
