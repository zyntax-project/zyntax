//! Type conversion from Whirlwind TypeExpression to TypedAST Type

use crate::error::{AdapterError, AdapterResult};
use whirlwind_ast::TypeExpression;
use zyntax_typed_ast::{
    AstArena, AsyncKind, CallingConvention, Lifetime, Mutability, NullabilityKind, ParamInfo,
    PrimitiveType, Type, TypeBound, TypeConstraint, TypeId, Variance,
};

/// Converts Whirlwind types to TypedAST types
pub struct TypeConverter {
    /// Type registry for managing named types
    /// Maps Whirlwind type names to TypedAST TypeIds
    type_registry: std::collections::HashMap<String, TypeId>,

    /// String arena for interning strings
    arena: AstArena,
}

impl TypeConverter {
    /// Create a new type converter
    pub fn new() -> Self {
        Self {
            type_registry: std::collections::HashMap::new(),
            arena: AstArena::new(),
        }
    }

    /// Convert a Whirlwind TypeExpression to TypedAST Type
    ///
    /// This is the main entry point for type conversion.
    ///
    /// # Whirlwind Type Mappings:
    ///
    /// - **Discrete** ("i32", "bool", etc.) → `Type::Primitive(...)`
    /// - **Union** (A | B) → `Type::Union([A, B])`
    /// - **Optional** (?T) → `Type::Optional(Box::new(T))`
    /// - **Functional** (fn(A) -> B) → `Type::Function { ... }`
    /// - **Array** ([T]) → `Type::Array { element_type: T, size: None }`
    /// - **Member** (core.io.Error) → `Type::Projection { ... }`
    /// - **Constraint** (T where ...) → Type with `TypeConstraint`
    /// - **Ternary** (conditional) → `DependentType::Conditional`
    /// - **This** → `Type::SelfType`
    pub fn convert_type(&mut self, whirlwind_type: &TypeExpression) -> AdapterResult<Type> {
        match whirlwind_type {
            TypeExpression::Discrete(discrete) => {
                let generic_args = discrete
                    .generic_args
                    .as_ref()
                    .map(|args| args.iter().map(|name| name.to_string()).collect::<Vec<_>>())
                    .unwrap_or_default();
                self.convert_discrete_type(&discrete.name.name, &generic_args)
            }
            TypeExpression::Union(union) => {
                let converted_types = union
                    .types
                    .iter()
                    .map(|t| self.convert_type(t))
                    .collect::<AdapterResult<Vec<_>>>()?;
                Ok(Type::Union(converted_types))
            }
            TypeExpression::Optional(maybe) => {
                let inner_type = self.convert_type(&maybe.value)?;
                Ok(Type::Optional(Box::new(inner_type)))
            }
            TypeExpression::Functional(func) => self.convert_function_type_expr(func),
            TypeExpression::Array(array) => {
                let elem = self.convert_type(&array.element_type)?;
                Ok(Type::Array {
                    element_type: Box::new(elem),
                    size: None, // Whirlwind uses dynamic arrays
                    nullability: NullabilityKind::NonNull,
                })
            }
            TypeExpression::Member(member) => self.convert_member_type(member),
            TypeExpression::This { .. } => Ok(Type::SelfType),
            TypeExpression::Constraint(constraint) => self.convert_constrained_type(constraint),
            TypeExpression::Ternary(ternary) => self.convert_ternary_type(ternary),
            TypeExpression::Invalid => {
                Err(AdapterError::type_conversion("Invalid type expression"))
            }
        }
    }

    /// Convert a discrete/named type (like "i32", "MyStruct<T>")
    fn convert_discrete_type(
        &mut self,
        name: &str,
        _generic_args: &[String],
    ) -> AdapterResult<Type> {
        // Map Whirlwind primitive type names to TypedAST primitives
        let primitive = match name {
            "i8" => Some(PrimitiveType::I8),
            "i16" => Some(PrimitiveType::I16),
            "i32" => Some(PrimitiveType::I32),
            "i64" => Some(PrimitiveType::I64),
            "i128" => Some(PrimitiveType::I128),
            "u8" => Some(PrimitiveType::U8),
            "u16" => Some(PrimitiveType::U16),
            "u32" => Some(PrimitiveType::U32),
            "u64" => Some(PrimitiveType::U64),
            "u128" => Some(PrimitiveType::U128),
            "f32" => Some(PrimitiveType::F32),
            "f64" => Some(PrimitiveType::F64),
            "bool" => Some(PrimitiveType::Bool),
            "char" => Some(PrimitiveType::Char),
            "string" | "str" => Some(PrimitiveType::String),
            "void" | "unit" => Some(PrimitiveType::Unit),
            _ => None,
        };

        if let Some(prim) = primitive {
            return Ok(Type::Primitive(prim));
        }

        // For non-primitive types, look up in registry
        // TODO: Implement named type lookup with generics
        Err(AdapterError::type_conversion(format!(
            "Named type '{}' not yet supported",
            name
        )))
    }

    /// Convert a Whirlwind FunctionalType to TypedAST Function type
    fn convert_function_type_expr(
        &mut self,
        func: &whirlwind_ast::FunctionalType,
    ) -> AdapterResult<Type> {
        let params: AdapterResult<Vec<ParamInfo>> = func
            .params
            .iter()
            .map(|param| {
                let param_type = if let Some(ref ty) = param.type_label {
                    self.convert_type(ty)?
                } else {
                    Type::Primitive(PrimitiveType::Unit) // Default to unit for untyped params
                };

                Ok(ParamInfo {
                    name: Some(self.arena.intern_string(&param.name.name)),
                    ty: param_type,
                    is_optional: param.is_optional,
                    is_varargs: false, // Whirlwind doesn't have explicit varargs in Parameter
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                })
            })
            .collect();

        let params = params?;

        let return_ty = if let Some(ref ret) = func.return_type {
            self.convert_type(ret)?
        } else {
            Type::Primitive(PrimitiveType::Unit)
        };

        Ok(Type::Function {
            params,
            return_type: Box::new(return_ty),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false, // Whirlwind handles defaults differently
            async_kind: AsyncKind::Sync, // TODO: Check if func has async marker
            calling_convention: CallingConvention::Default,
            nullability: NullabilityKind::NonNull,
        })
    }

    /// Convert a Whirlwind MemberType (e.g., core.io.Error)
    fn convert_member_type(&mut self, member: &whirlwind_ast::MemberType) -> AdapterResult<Type> {
        // For now, treat as a named type (simplified)
        // TODO: Implement proper module path resolution
        Err(AdapterError::unsupported(
            "Member type conversion not yet fully implemented",
        ))
    }

    /// Convert a constrained type (e.g., T where T implements Default)
    fn convert_constrained_type(
        &mut self,
        constraint: &whirlwind_ast::BoundConstraintType,
    ) -> AdapterResult<Type> {
        // Convert the base type
        let base_type = self.convert_discrete_type(&constraint.consequent.name.name, &[])?;

        // TODO: Convert the constraint clause to TypeConstraint
        // For now, return the base type
        Ok(base_type)
    }

    /// Convert a ternary/conditional type (e.g., if T implements Default String else Bool)
    fn convert_ternary_type(
        &mut self,
        ternary: &whirlwind_ast::TernaryType,
    ) -> AdapterResult<Type> {
        // TODO: Implement dependent type conversion
        // For now, return the consequent type
        self.convert_type(&ternary.consequent)
    }

    /// Register a named type in the registry
    pub fn register_type(&mut self, name: String, type_id: TypeId) {
        self.type_registry.insert(name, type_id);
    }

    /// Look up a type by name
    pub fn lookup_type(&self, name: &str) -> Option<TypeId> {
        self.type_registry.get(name).copied()
    }
}

impl Default for TypeConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_type_conversion() {
        let mut converter = TypeConverter::new();

        // Test basic primitive mapping
        let i32_type = converter.convert_discrete_type("i32", &[]).unwrap();
        assert!(matches!(i32_type, Type::Primitive(PrimitiveType::I32)));

        let bool_type = converter.convert_discrete_type("bool", &[]).unwrap();
        assert!(matches!(bool_type, Type::Primitive(PrimitiveType::Bool)));
    }
}
