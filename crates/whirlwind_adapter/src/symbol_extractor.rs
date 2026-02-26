//! Helper functions to extract type and signature information from Whirlwind's SymbolLibrary

use crate::error::{AdapterError, AdapterResult};
use crate::type_converter::TypeConverter;
use whirlwind_analyzer::{
    EvaluatedType, IntermediateType, SemanticSymbol, SemanticSymbolKind, SymbolIndex, SymbolLibrary,
};
use zyntax_typed_ast::{
    typed_ast::{
        ParameterAttribute, ParameterKind, TypedMethodParam, TypedParameter, TypedTypeBound,
        TypedVariantFields,
    },
    AstArena, AsyncKind, CallingConvention, InternedString, Mutability, NullabilityKind, ParamInfo,
    PrimitiveType, Type, TypeRegistry, TypeVar, TypeVarId, TypeVarKind, Visibility,
};

pub struct SymbolExtractor {
    type_converter: TypeConverter,
}

impl SymbolExtractor {
    pub fn new() -> Self {
        Self {
            type_converter: TypeConverter::new(),
        }
    }

    /// Extract the type of a variable from its symbol
    pub fn extract_variable_type(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Type> {
        match &symbol.kind {
            SemanticSymbolKind::Variable {
                declared_type,
                inferred_type,
                ..
            } => {
                // Prefer declared type, fall back to inferred
                if let Some(declared) = declared_type {
                    self.convert_intermediate_type(declared, symbol_library, type_registry, arena)
                } else {
                    self.convert_evaluated_type(inferred_type, symbol_library, type_registry, arena)
                }
            }
            SemanticSymbolKind::Attribute { declared_type, .. } => {
                self.convert_intermediate_type(declared_type, symbol_library, type_registry, arena)
            }
            _ => Ok(Type::Unknown),
        }
    }

    /// Extract function signature: parameters and return type
    pub fn extract_function_signature(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<(Vec<TypedParameter>, Type)> {
        match &symbol.kind {
            SemanticSymbolKind::Function {
                params,
                return_type,
                ..
            } => {
                let typed_params =
                    self.extract_parameters(params, symbol_library, type_registry, arena)?;
                let return_ty = if let Some(ret_type) = return_type {
                    self.convert_intermediate_type(ret_type, symbol_library, type_registry, arena)?
                } else {
                    Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)
                };
                Ok((typed_params, return_ty))
            }
            _ => Ok((vec![], Type::Unknown)),
        }
    }

    /// Extract method signature: parameters and return type
    pub fn extract_method_signature(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<(Vec<TypedMethodParam>, Type, bool, bool)> {
        match &symbol.kind {
            SemanticSymbolKind::Method {
                params,
                return_type,
                is_static,
                is_async,
                ..
            } => {
                let typed_params =
                    self.extract_method_parameters(params, symbol_library, type_registry, arena)?;
                let return_ty = if let Some(ret_type) = return_type {
                    self.convert_intermediate_type(ret_type, symbol_library, type_registry, arena)?
                } else {
                    Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)
                };
                Ok((typed_params, return_ty, *is_static, *is_async))
            }
            _ => Ok((vec![], Type::Unknown, false, false)),
        }
    }

    /// Extract enum variants
    pub fn extract_enum_variants(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Vec<zyntax_typed_ast::typed_ast::TypedVariant>> {
        match &symbol.kind {
            SemanticSymbolKind::Enum { variants, .. } => {
                let mut typed_variants = Vec::new();

                for variant_idx in variants {
                    if let Some(variant_symbol) = symbol_library.get(*variant_idx) {
                        let variant_name = arena.intern_string(&variant_symbol.name);

                        // Extract associated types if any
                        let variant_fields =
                            if let SemanticSymbolKind::Variant { tagged_types, .. } =
                                &variant_symbol.kind
                            {
                                if tagged_types.is_empty() {
                                    TypedVariantFields::Unit
                                } else {
                                    let types = tagged_types
                                        .iter()
                                        .map(|ty| {
                                            self.convert_intermediate_type(
                                                ty,
                                                symbol_library,
                                                type_registry,
                                                arena,
                                            )
                                        })
                                        .collect::<AdapterResult<Vec<_>>>()?;
                                    TypedVariantFields::Tuple(types)
                                }
                            } else {
                                TypedVariantFields::Unit
                            };

                        typed_variants.push(zyntax_typed_ast::typed_ast::TypedVariant {
                            name: variant_name,
                            fields: variant_fields,
                            discriminant: None,
                            span: zyntax_typed_ast::Span::default(),
                        });
                    }
                }

                Ok(typed_variants)
            }
            _ => Ok(vec![]),
        }
    }

    /// Extract type alias target
    pub fn extract_type_alias_target(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Type> {
        match &symbol.kind {
            SemanticSymbolKind::TypeName { value, .. } => {
                self.convert_intermediate_type(value, symbol_library, type_registry, arena)
            }
            _ => Ok(Type::Unknown),
        }
    }

    /// Extract visibility from symbol
    pub fn extract_visibility(symbol: &SemanticSymbol) -> Visibility {
        match &symbol.kind {
            SemanticSymbolKind::Function { is_public, .. }
            | SemanticSymbolKind::Variable { is_public, .. }
            | SemanticSymbolKind::Model { is_public, .. }
            | SemanticSymbolKind::Interface { is_public, .. }
            | SemanticSymbolKind::Enum { is_public, .. }
            | SemanticSymbolKind::TypeName { is_public, .. }
            | SemanticSymbolKind::Method { is_public, .. }
            | SemanticSymbolKind::Attribute { is_public, .. } => {
                if *is_public {
                    Visibility::Public
                } else {
                    Visibility::Private
                }
            }
            _ => Visibility::Private,
        }
    }

    /// Extract generic parameters from symbol
    pub fn extract_generic_params(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Vec<zyntax_typed_ast::typed_ast::TypedTypeParam>> {
        let generic_param_indices = match &symbol.kind {
            SemanticSymbolKind::Function { generic_params, .. }
            | SemanticSymbolKind::Model { generic_params, .. }
            | SemanticSymbolKind::Interface { generic_params, .. }
            | SemanticSymbolKind::Enum { generic_params, .. }
            | SemanticSymbolKind::TypeName { generic_params, .. }
            | SemanticSymbolKind::Method { generic_params, .. } => generic_params,
            _ => return Ok(vec![]),
        };

        let mut type_params = Vec::new();
        for param_idx in generic_param_indices {
            if let Some(param_symbol) = symbol_library.get(*param_idx) {
                let param_name = arena.intern_string(&param_symbol.name);

                // Extract bounds if any
                let bounds = if let SemanticSymbolKind::GenericParameter { interfaces, .. } =
                    &param_symbol.kind
                {
                    interfaces
                        .iter()
                        .map(|ty| {
                            self.convert_intermediate_type(ty, symbol_library, type_registry, arena)
                                .map(|t| TypedTypeBound::Trait(t))
                        })
                        .collect::<AdapterResult<Vec<_>>>()?
                } else {
                    vec![]
                };

                type_params.push(zyntax_typed_ast::typed_ast::TypedTypeParam {
                    name: param_name,
                    bounds,
                    default: None,
                    span: zyntax_typed_ast::Span::default(),
                });
            }
        }

        Ok(type_params)
    }

    /// Extract model base class and interfaces
    pub fn extract_model_inheritance(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<(Option<Type>, Vec<Type>)> {
        match &symbol.kind {
            SemanticSymbolKind::Model { interfaces, .. } => {
                let implements = interfaces
                    .iter()
                    .map(|ty| {
                        self.convert_intermediate_type(ty, symbol_library, type_registry, arena)
                    })
                    .collect::<AdapterResult<Vec<_>>>()?;
                Ok((None, implements)) // Whirlwind doesn't have base classes, only interfaces
            }
            _ => Ok((None, vec![])),
        }
    }

    /// Extract interface parents
    pub fn extract_interface_extends(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Vec<Type>> {
        match &symbol.kind {
            SemanticSymbolKind::Interface { interfaces, .. } => interfaces
                .iter()
                .map(|ty| self.convert_intermediate_type(ty, symbol_library, type_registry, arena))
                .collect::<AdapterResult<Vec<_>>>(),
            _ => Ok(vec![]),
        }
    }

    /// Extract constructor parameters from a model
    pub fn extract_constructor_params(
        &mut self,
        symbol: &SemanticSymbol,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Option<Vec<TypedMethodParam>>> {
        match &symbol.kind {
            SemanticSymbolKind::Model {
                constructor_parameters,
                ..
            } => {
                if let Some(param_indices) = constructor_parameters {
                    let params = self.extract_method_parameters(
                        param_indices,
                        symbol_library,
                        type_registry,
                        arena,
                    )?;
                    Ok(Some(params))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    // Private helper methods

    fn extract_parameters(
        &mut self,
        param_indices: &[SymbolIndex],
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Vec<TypedParameter>> {
        let mut params = Vec::new();

        for param_idx in param_indices {
            if let Some(param_symbol) = symbol_library.get(*param_idx) {
                let param_name = arena.intern_string(&param_symbol.name);

                let param_type = if let SemanticSymbolKind::Parameter {
                    param_type,
                    inferred_type,
                    ..
                } = &param_symbol.kind
                {
                    if let Some(declared) = param_type {
                        self.convert_intermediate_type(
                            declared,
                            symbol_library,
                            type_registry,
                            arena,
                        )?
                    } else {
                        self.convert_evaluated_type(
                            inferred_type,
                            symbol_library,
                            type_registry,
                            arena,
                        )?
                    }
                } else {
                    Type::Unknown
                };

                params.push(TypedParameter {
                    name: param_name,
                    ty: param_type,
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: zyntax_typed_ast::Span::default(),
                });
            }
        }

        Ok(params)
    }

    fn extract_method_parameters(
        &mut self,
        param_indices: &[SymbolIndex],
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Vec<TypedMethodParam>> {
        let mut params = Vec::new();

        for param_idx in param_indices {
            if let Some(param_symbol) = symbol_library.get(*param_idx) {
                let param_name = arena.intern_string(&param_symbol.name);

                let (param_type, is_optional) = if let SemanticSymbolKind::Parameter {
                    param_type,
                    inferred_type,
                    is_optional,
                    ..
                } = &param_symbol.kind
                {
                    let ty = if let Some(declared) = param_type {
                        self.convert_intermediate_type(
                            declared,
                            symbol_library,
                            type_registry,
                            arena,
                        )?
                    } else {
                        self.convert_evaluated_type(
                            inferred_type,
                            symbol_library,
                            type_registry,
                            arena,
                        )?
                    };
                    (ty, *is_optional)
                } else {
                    (Type::Unknown, false)
                };

                params.push(TypedMethodParam {
                    name: param_name,
                    ty: param_type,
                    mutability: Mutability::Immutable,
                    is_self: false,               // TODO: Detect self parameter
                    kind: ParameterKind::Regular, // TODO: Handle optional parameters
                    default_value: None,
                    attributes: vec![],
                    span: zyntax_typed_ast::Span::default(),
                });
            }
        }

        Ok(params)
    }

    pub fn convert_intermediate_type(
        &mut self,
        int_type: &IntermediateType,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Type> {
        use whirlwind_analyzer::IntermediateType;

        match int_type {
            IntermediateType::SimpleType {
                value,
                generic_args,
                ..
            } => {
                // Get the symbol from the symbol library
                if let Some(symbol) = symbol_library.get(*value) {
                    let type_name = arena.intern_string(&symbol.name);

                    // Check if it's a primitive type first
                    if let Some(primitive_type) = self.map_to_primitive(&symbol.name) {
                        return Ok(Type::Primitive(primitive_type));
                    }

                    // Look up in type registry
                    if let Some(type_def) = type_registry.get_type_by_name(type_name) {
                        // Convert generic args if any
                        let type_args = generic_args
                            .iter()
                            .map(|arg| {
                                self.convert_intermediate_type(
                                    arg,
                                    symbol_library,
                                    type_registry,
                                    arena,
                                )
                            })
                            .collect::<AdapterResult<Vec<_>>>()?;

                        return Ok(Type::Named {
                            id: type_def.id,
                            type_args,
                            const_args: vec![],
                            variance: vec![],
                            nullability: NullabilityKind::NonNull,
                        });
                    }

                    // If not found in registry, create a type variable placeholder
                    return Ok(Type::TypeVar(TypeVar {
                        id: TypeVarId::next(),
                        name: Some(type_name),
                        kind: TypeVarKind::Type,
                    }));
                }
                Ok(Type::Unknown)
            }

            IntermediateType::FunctionType {
                params,
                return_type,
                ..
            } => {
                let param_infos = params
                    .iter()
                    .map(|p| {
                        let param_ty = if let Some(ref type_label) = p.type_label {
                            self.convert_intermediate_type(
                                type_label,
                                symbol_library,
                                type_registry,
                                arena,
                            )?
                        } else {
                            Type::Unknown
                        };
                        Ok(ParamInfo {
                            name: Some(arena.intern_string(&p.name)),
                            ty: param_ty,
                            is_optional: p.is_optional,
                            is_varargs: false,
                            is_keyword_only: false,
                            is_positional_only: false,
                            is_out: false,
                            is_ref: false,
                            is_inout: false,
                        })
                    })
                    .collect::<AdapterResult<Vec<_>>>()?;

                let ret_type = if let Some(ret) = return_type {
                    self.convert_intermediate_type(ret, symbol_library, type_registry, arena)?
                } else {
                    Type::Primitive(PrimitiveType::Unit)
                };

                Ok(Type::Function {
                    params: param_infos,
                    return_type: Box::new(ret_type),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: AsyncKind::Sync,
                    calling_convention: CallingConvention::Default,
                    nullability: NullabilityKind::NonNull,
                })
            }

            IntermediateType::ArrayType { element_type, .. } => {
                let elem_ty = self.convert_intermediate_type(
                    element_type,
                    symbol_library,
                    type_registry,
                    arena,
                )?;
                Ok(Type::Array {
                    element_type: Box::new(elem_ty),
                    size: None,
                    nullability: NullabilityKind::NonNull,
                })
            }

            IntermediateType::MaybeType { value, .. } => {
                let inner_ty =
                    self.convert_intermediate_type(value, symbol_library, type_registry, arena)?;
                // MaybeType represents optional/nullable types - wrap in Optional
                Ok(Type::Optional(Box::new(inner_ty)))
            }

            IntermediateType::UnionType { types, .. } => {
                let variant_types = types
                    .iter()
                    .map(|t| {
                        self.convert_intermediate_type(t, symbol_library, type_registry, arena)
                    })
                    .collect::<AdapterResult<Vec<_>>>()?;

                Ok(Type::Union(variant_types))
            }

            IntermediateType::This { meaning, .. } => {
                if let Some(symbol_idx) = meaning {
                    if let Some(symbol) = symbol_library.get(*symbol_idx) {
                        let type_name = arena.intern_string(&symbol.name);
                        if let Some(type_def) = type_registry.get_type_by_name(type_name) {
                            return Ok(Type::Named {
                                id: type_def.id,
                                type_args: vec![],
                                const_args: vec![],
                                variance: vec![],
                                nullability: NullabilityKind::NonNull,
                            });
                        }
                    }
                }
                Ok(Type::SelfType)
            }

            IntermediateType::Placeholder => Ok(Type::Unknown),

            IntermediateType::MemberType { .. } => {
                // Member types like "Module.Type" - for now, return Unknown
                // TODO: Implement member type resolution
                Ok(Type::Unknown)
            }

            IntermediateType::TernaryType { .. } => {
                // Conditional types - not supported yet
                Ok(Type::Unknown)
            }

            IntermediateType::BoundConstraintType { consequent, .. } => {
                // For constraint types, just convert the consequent
                self.convert_intermediate_type(consequent, symbol_library, type_registry, arena)
            }
        }
    }

    /// Map Whirlwind type names to Zyntax primitive types
    fn map_to_primitive(&self, type_name: &str) -> Option<PrimitiveType> {
        match type_name {
            "int" | "Integer" => Some(PrimitiveType::I32),
            "float" | "Float" => Some(PrimitiveType::F64),
            "bool" | "Boolean" => Some(PrimitiveType::Bool),
            "string" | "String" => Some(PrimitiveType::String),
            "char" | "Char" => Some(PrimitiveType::Char),
            "void" | "Void" => Some(PrimitiveType::Unit),
            _ => None,
        }
    }

    fn convert_evaluated_type(
        &mut self,
        eval_type: &EvaluatedType,
        symbol_library: &SymbolLibrary,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<Type> {
        match eval_type {
            EvaluatedType::Void => Ok(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)),
            EvaluatedType::Never => Ok(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)),
            EvaluatedType::Unknown => Ok(Type::Unknown),
            EvaluatedType::Partial { types } => {
                let converted_types: Vec<_> = types
                    .iter()
                    .map(|t| self.convert_evaluated_type(t, symbol_library, type_registry, arena))
                    .collect::<AdapterResult<_>>()?;
                Ok(Type::Union(converted_types))
            }
            // TODO: Implement full EvaluatedType conversion
            _ => Ok(Type::Unknown),
        }
    }
}

impl Default for SymbolExtractor {
    fn default() -> Self {
        Self::new()
    }
}
