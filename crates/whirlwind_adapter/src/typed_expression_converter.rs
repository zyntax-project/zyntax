//! Converter for Whirlwind's TypedExpression (analyzed expressions with inferred types)

use crate::error::{AdapterError, AdapterResult};
use crate::symbol_extractor::SymbolExtractor;
use crate::type_converter::TypeConverter;
use whirlwind_analyzer::{
    EvaluatedType, Literal, LiteralMap, SymbolLibrary, TypedAccessExpr, TypedArrayExpr,
    TypedAssignmentExpr, TypedBinExpr, TypedBlock, TypedCallExpr, TypedExpression, TypedFnExpr,
    TypedIdent, TypedIfExpr, TypedIndexExpr, TypedLogicExpr, TypedStmnt, TypedUnaryExpr,
    TypedUpdateExpr,
};
use whirlwind_ast::{AssignOperator, BinOperator, LogicOperator, UnaryOperator, UpdateOperator};
use zyntax_typed_ast::{
    typed_ast::{
        TypedBinary, TypedBlock as ZyntaxTypedBlock, TypedCall, TypedFieldAccess, TypedFor,
        TypedIfExpr as ZyntaxTypedIfExpr, TypedIndex, TypedLambda, TypedLambdaBody,
        TypedLambdaParam, TypedLet, TypedLiteral, TypedPattern, TypedUnary,
    },
    AstArena, BinaryOp, InternedString, Mutability, Span, Type, TypeRegistry,
    TypedExpression as ZyntaxTypedExpression, TypedNode, TypedStatement, UnaryOp, Visibility,
};

/// Converts Whirlwind's TypedExpression (with inferred types) to Zyntax TypedExpression
pub struct TypedExpressionConverter {
    type_converter: TypeConverter,
}

impl TypedExpressionConverter {
    pub fn new() -> Self {
        Self {
            type_converter: TypeConverter::new(),
        }
    }

    /// Convert a Whirlwind TypedExpression to our TypedExpression
    pub fn convert_typed_expression(
        &mut self,
        expr: &TypedExpression,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        match expr {
            TypedExpression::Identifier(ident) => self.convert_identifier(
                ident,
                symbol_library,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::Literal(literal_idx) => {
                self.convert_literal(*literal_idx, literals, arena)
            }
            TypedExpression::ThisExpr(this_expr) => self.convert_this_expr(
                this_expr,
                symbol_library,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::CallExpr(call_expr) => self.convert_call_expr(
                call_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::FnExpr(fn_expr) => self.convert_fn_expr(
                fn_expr,
                symbol_library,
                literals,
                arena,
                symbol_extractor,
                type_registry,
            ),
            TypedExpression::Block(block) => self.convert_block_expr(
                block,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::IfExpr(if_expr) => self.convert_if_expr(
                if_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::AccessExpr(access_expr) => self.convert_access_expr(
                access_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::ArrayExpr(array_expr) => self.convert_array_expr(
                array_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::IndexExpr(index_expr) => self.convert_index_expr(
                index_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::BinaryExpr(bin_expr) => self.convert_binary_expr(
                bin_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::AssignmentExpr(assign_expr) => self.convert_assignment_expr(
                assign_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::UnaryExpr(unary_expr) => self.convert_unary_expr(
                unary_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::LogicExpr(logic_expr) => self.convert_logic_expr(
                logic_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
            TypedExpression::UpdateExpr(update_expr) => self.convert_update_expr(
                update_expr,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            ),
        }
    }

    fn convert_identifier(
        &mut self,
        ident: &TypedIdent,
        symbol_library: &SymbolLibrary,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        use whirlwind_analyzer::SemanticSymbolKind;

        let symbol = symbol_library.get(ident.value).ok_or_else(|| {
            AdapterError::expression_conversion("Symbol not found for identifier")
        })?;

        let var_name = arena.intern_string(&symbol.name);

        // Extract type from symbol's declared or inferred type
        let ty = match &symbol.kind {
            SemanticSymbolKind::Parameter { param_type, .. } => {
                if let Some(declared) = param_type {
                    // Convert the declared type using symbol_extractor
                    symbol_extractor.convert_intermediate_type(
                        declared,
                        symbol_library,
                        type_registry,
                        arena,
                    )?
                } else {
                    Type::Unknown
                }
            }
            SemanticSymbolKind::Variable { declared_type, .. } => {
                if let Some(declared) = declared_type {
                    symbol_extractor.convert_intermediate_type(
                        declared,
                        symbol_library,
                        type_registry,
                        arena,
                    )?
                } else {
                    Type::Unknown
                }
            }
            SemanticSymbolKind::Function {
                params,
                return_type,
                is_async,
                ..
            } => {
                use zyntax_typed_ast::{
                    AsyncKind, CallingConvention, NullabilityKind, ParamInfo, PrimitiveType,
                };

                // Extract function signature
                let param_infos: Vec<ParamInfo> = params
                    .iter()
                    .filter_map(|param_idx| {
                        symbol_library.get(*param_idx).and_then(|param_sym| {
                            let param_name = arena.intern_string(&param_sym.name);
                            if let SemanticSymbolKind::Parameter { param_type, .. } =
                                &param_sym.kind
                            {
                                param_type.as_ref().and_then(|pt| {
                                    symbol_extractor
                                        .convert_intermediate_type(
                                            pt,
                                            symbol_library,
                                            type_registry,
                                            arena,
                                        )
                                        .ok()
                                        .map(|ty| ParamInfo {
                                            name: Some(param_name),
                                            ty,
                                            is_optional: false,
                                            is_varargs: false,
                                            is_keyword_only: false,
                                            is_positional_only: false,
                                            is_out: false,
                                            is_ref: false,
                                            is_inout: false,
                                        })
                                })
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                let ret_type = if let Some(ret_ty) = return_type {
                    symbol_extractor.convert_intermediate_type(
                        ret_ty,
                        symbol_library,
                        type_registry,
                        arena,
                    )?
                } else {
                    Type::Primitive(PrimitiveType::Unit)
                };

                Type::Function {
                    params: param_infos,
                    return_type: Box::new(ret_type),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: if *is_async {
                        AsyncKind::Async
                    } else {
                        AsyncKind::Sync
                    },
                    calling_convention: CallingConvention::Default,
                    nullability: NullabilityKind::NonNull,
                }
            }
            _ => Type::Unknown,
        };

        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Variable(var_name),
            ty,
            span,
        ))
    }

    fn convert_literal(
        &mut self,
        literal_idx: whirlwind_analyzer::LiteralIndex,
        literals: &LiteralMap,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let literal = literals
            .get(literal_idx)
            .ok_or_else(|| AdapterError::expression_conversion("Literal not found in map"))?;

        match literal {
            Literal::StringLiteral { value, .. } => {
                let string_value = arena.intern_string(&value.value);
                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::String);
                let span = Span::default();
                Ok(TypedNode::new(
                    ZyntaxTypedExpression::Literal(TypedLiteral::String(string_value)),
                    ty,
                    span,
                ))
            }
            Literal::NumericLiteral { value, .. } => {
                // Whirlwind's Number is an enum, need to handle it properly
                let numeric_str = format!("{:?}", value.value); // Convert enum to string representation

                // Try to parse as integer first, then float
                if let Ok(int_val) = numeric_str.parse::<i128>() {
                    let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::I64);
                    let span = Span::default();
                    Ok(TypedNode::new(
                        ZyntaxTypedExpression::Literal(TypedLiteral::Integer(int_val)),
                        ty,
                        span,
                    ))
                } else if let Ok(float_val) = numeric_str.parse::<f64>() {
                    let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::F64);
                    let span = Span::default();
                    Ok(TypedNode::new(
                        ZyntaxTypedExpression::Literal(TypedLiteral::Float(float_val)),
                        ty,
                        span,
                    ))
                } else {
                    // Default to 0
                    let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::I64);
                    let span = Span::default();
                    Ok(TypedNode::new(
                        ZyntaxTypedExpression::Literal(TypedLiteral::Integer(0)),
                        ty,
                        span,
                    ))
                }
            }
            Literal::BooleanLiteral { value, .. } => {
                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool);
                let span = Span::default();
                Ok(TypedNode::new(
                    ZyntaxTypedExpression::Literal(TypedLiteral::Bool(*value)),
                    ty,
                    span,
                ))
            }
        }
    }

    fn convert_this_expr(
        &mut self,
        this_expr: &whirlwind_analyzer::TypedThisExpr,
        symbol_library: &SymbolLibrary,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        use whirlwind_analyzer::SemanticSymbolKind;

        // Convert 'this' to a variable reference called "this"
        let this_name = arena.intern_string("this");

        // Try to extract type from model_or_interface if available
        let ty = if let Some(model_idx) = this_expr.model_or_interface {
            if let Some(model_symbol) = symbol_library.get(model_idx) {
                // Get the model/class name and look it up in the type registry
                let class_name = arena.intern_string(&model_symbol.name);

                // Check if we have this type registered
                if let Some(type_def) = type_registry.get_type_by_name(class_name) {
                    // Create a Type::Named instance for this class
                    type_registry.make_type(type_def.id, vec![])
                } else {
                    // Type not yet registered, fall back to Unknown
                    self.convert_evaluated_type(&this_expr.inferred_type, symbol_library)?
                }
            } else {
                self.convert_evaluated_type(&this_expr.inferred_type, symbol_library)?
            }
        } else {
            self.convert_evaluated_type(&this_expr.inferred_type, symbol_library)?
        };

        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Variable(this_name),
            ty,
            span,
        ))
    }

    fn convert_call_expr(
        &mut self,
        call_expr: &TypedCallExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let caller = self.convert_typed_expression(
            &call_expr.caller,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let arguments: Vec<_> = call_expr
            .arguments
            .iter()
            .map(|arg| {
                self.convert_typed_expression(
                    arg,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )
            })
            .collect::<AdapterResult<_>>()?;

        // Infer call result type from callee's function type
        let ty = match &caller.ty {
            Type::Function { return_type, .. } => (**return_type).clone(),
            _ => {
                // Fall back to Whirlwind's inferred type if callee is not a function
                self.convert_evaluated_type(&call_expr.inferred_type, symbol_library)?
            }
        };
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Call(TypedCall {
                callee: Box::new(caller),
                positional_args: arguments,
                named_args: vec![],
                type_args: vec![],
            }),
            ty,
            span,
        ))
    }

    fn convert_fn_expr(
        &mut self,
        fn_expr: &TypedFnExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        arena: &mut AstArena,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        // Convert parameters
        let params: Vec<TypedLambdaParam> = fn_expr
            .params
            .iter()
            .map(|param_idx| {
                let symbol = symbol_library.get(*param_idx).ok_or_else(|| {
                    AdapterError::expression_conversion("Parameter symbol not found")
                })?;
                let name = arena.intern_string(&symbol.name);
                // TODO: Extract actual parameter type from symbol
                Ok(TypedLambdaParam {
                    name,
                    ty: Some(Type::Unknown),
                })
            })
            .collect::<AdapterResult<Vec<TypedLambdaParam>>>()?;

        // Convert body
        let body = self.convert_typed_expression(
            &fn_expr.body,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let ty = self.convert_evaluated_type(&fn_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Lambda(TypedLambda {
                params,
                body: TypedLambdaBody::Expression(Box::new(body)),
                captures: vec![],
            }),
            ty,
            span,
        ))
    }

    fn convert_block_expr(
        &mut self,
        block: &TypedBlock,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        // Convert block to a sequence of statements
        // Since we don't have a Block variant, we'll convert the last expression
        // and wrap intermediate statements somehow

        if block.statements.is_empty() {
            // Empty block returns unit
            let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
            let span = Span::default();
            return Ok(TypedNode::new(
                ZyntaxTypedExpression::Literal(TypedLiteral::Integer(0)), // Placeholder
                ty,
                span,
            ));
        }

        // For now, just convert the last statement as an expression
        // TODO: Properly handle block expressions with multiple statements
        let last_stmt = &block.statements[block.statements.len() - 1];

        // Try to extract expression from last statement
        match last_stmt {
            TypedStmnt::FreeExpression(expr) | TypedStmnt::ExpressionStatement(expr) => self
                .convert_typed_expression(
                    expr,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                ),
            TypedStmnt::ReturnStatement(ret) => {
                if let Some(ref expr) = ret.value {
                    self.convert_typed_expression(
                        expr,
                        symbol_library,
                        literals,
                        symbol_extractor,
                        type_registry,
                        arena,
                    )
                } else {
                    let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                    let span = Span::default();
                    Ok(TypedNode::new(
                        ZyntaxTypedExpression::Literal(TypedLiteral::Integer(0)),
                        ty,
                        span,
                    ))
                }
            }
            _ => {
                // Non-expression statement, return unit
                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                let span = Span::default();
                Ok(TypedNode::new(
                    ZyntaxTypedExpression::Literal(TypedLiteral::Integer(0)),
                    ty,
                    span,
                ))
            }
        }
    }

    fn convert_if_expr(
        &mut self,
        if_expr: &TypedIfExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let condition = self.convert_typed_expression(
            &if_expr.condition,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        // Convert the then block to an expression
        let then_branch = self.convert_block_expr(
            &if_expr.consequent,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let else_branch = if let Some(alternate) = &if_expr.alternate {
            // alternate.expression could be another if or a block
            self.convert_typed_expression(
                &alternate.expression,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            )?
        } else {
            // No else branch - create a unit literal
            let span = Span::default();
            TypedNode::new(
                ZyntaxTypedExpression::Literal(TypedLiteral::Integer(0)), // Placeholder for unit
                Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                span,
            )
        };

        let ty = self.convert_evaluated_type(&if_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::If(ZyntaxTypedIfExpr {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            }),
            ty,
            span,
        ))
    }

    fn convert_access_expr(
        &mut self,
        access_expr: &TypedAccessExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        use whirlwind_analyzer::SemanticSymbolKind;

        let object = self.convert_typed_expression(
            &access_expr.object,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        // The property needs to be extracted as a field name and its type
        let (field_name, field_type) = match &access_expr.property {
            TypedExpression::Identifier(ident) => {
                let symbol = symbol_library.get(ident.value).ok_or_else(|| {
                    AdapterError::expression_conversion("Property symbol not found")
                })?;
                let name = arena.intern_string(&symbol.name);

                // Try to extract the field's type from the symbol
                let ty = match &symbol.kind {
                    SemanticSymbolKind::Attribute { declared_type, .. } => {
                        // Extract the field's declared type
                        symbol_extractor.convert_intermediate_type(
                            declared_type,
                            symbol_library,
                            type_registry,
                            arena,
                        )?
                    }
                    SemanticSymbolKind::Property { resolved, .. } => {
                        // Property that may or may not be resolved
                        if let Some(resolved_idx) = resolved {
                            // Try to get the resolved symbol
                            if let Some(resolved_symbol) = symbol_library.get(*resolved_idx) {
                                // Recursively extract the type from the resolved symbol
                                match &resolved_symbol.kind {
                                    SemanticSymbolKind::Attribute { declared_type, .. } => {
                                        symbol_extractor.convert_intermediate_type(
                                            declared_type,
                                            symbol_library,
                                            type_registry,
                                            arena,
                                        )?
                                    }
                                    _ => {
                                        // Fall back to inferred type
                                        self.convert_evaluated_type(
                                            &access_expr.inferred_type,
                                            symbol_library,
                                        )?
                                    }
                                }
                            } else {
                                self.convert_evaluated_type(
                                    &access_expr.inferred_type,
                                    symbol_library,
                                )?
                            }
                        } else {
                            // Property not resolved yet, need to resolve it manually
                            // Look up the field in the object's class definition
                            if let Type::Named { id, .. } = &object.ty {
                                if let Some(type_def) = type_registry.get_type_by_id(*id) {
                                    // Find the field with matching name
                                    if let Some(field) =
                                        type_def.fields.iter().find(|f| f.name == name)
                                    {
                                        field.ty.clone()
                                    } else {
                                        self.convert_evaluated_type(
                                            &access_expr.inferred_type,
                                            symbol_library,
                                        )?
                                    }
                                } else {
                                    self.convert_evaluated_type(
                                        &access_expr.inferred_type,
                                        symbol_library,
                                    )?
                                }
                            } else {
                                self.convert_evaluated_type(
                                    &access_expr.inferred_type,
                                    symbol_library,
                                )?
                            }
                        }
                    }
                    _ => {
                        // Not a field/attribute, use inferred type
                        self.convert_evaluated_type(&access_expr.inferred_type, symbol_library)?
                    }
                };

                (name, ty)
            }
            _ => {
                return Err(AdapterError::expression_conversion(
                    "Expected identifier for field access",
                ));
            }
        };

        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: field_name,
            }),
            field_type,
            span,
        ))
    }

    fn convert_array_expr(
        &mut self,
        array_expr: &TypedArrayExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let elements: Vec<_> = array_expr
            .elements
            .iter()
            .map(|elem| {
                self.convert_typed_expression(
                    elem,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )
            })
            .collect::<AdapterResult<_>>()?;

        let ty = self.convert_evaluated_type(&array_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Array(elements),
            ty,
            span,
        ))
    }

    fn convert_index_expr(
        &mut self,
        index_expr: &TypedIndexExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let object = self.convert_typed_expression(
            &index_expr.object,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;
        let index = self.convert_typed_expression(
            &index_expr.index,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let ty = self.convert_evaluated_type(&index_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Index(TypedIndex {
                object: Box::new(object),
                index: Box::new(index),
            }),
            ty,
            span,
        ))
    }

    fn convert_binary_expr(
        &mut self,
        bin_expr: &TypedBinExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,

        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let left = self.convert_typed_expression(
            &bin_expr.left,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;
        let right = self.convert_typed_expression(
            &bin_expr.right,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let op = self.convert_binary_operator(&bin_expr.operator)?;

        // Infer result type based on operator
        let ty = match op {
            // Comparison operators always return Bool
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool),
            // Arithmetic operators return the type of operands (if both are same)
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                // If both operands have the same type, use that type
                if left.ty == right.ty {
                    left.ty.clone()
                } else {
                    // Fall back to Whirlwind's inference if types don't match
                    self.convert_evaluated_type(&bin_expr.inferred_type, symbol_library)?
                }
            }
            // Logical operators return Bool
            BinaryOp::And | BinaryOp::Or => Type::Primitive(zyntax_typed_ast::PrimitiveType::Bool),
            // Bitwise operators return the type of operands
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => {
                if left.ty == right.ty {
                    left.ty.clone()
                } else {
                    self.convert_evaluated_type(&bin_expr.inferred_type, symbol_library)?
                }
            }
            // Assignment returns the type of the right side (the value being assigned)
            BinaryOp::Assign => right.ty.clone(),
            // Zig error handling operators - return the type of the right side (default value)
            BinaryOp::Orelse | BinaryOp::Catch => right.ty.clone(),
        };
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Binary(TypedBinary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }),
            ty,
            span,
        ))
    }

    fn convert_assignment_expr(
        &mut self,
        assign_expr: &TypedAssignmentExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let left = self.convert_typed_expression(
            &assign_expr.left,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;
        let right = self.convert_typed_expression(
            &assign_expr.right,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        // Assignment expressions are converted to binary operations
        // For compound assignments like +=, we extract the operation
        let op = self.convert_assign_operator(&assign_expr.operator)?;
        let ty = self.convert_evaluated_type(&assign_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Binary(TypedBinary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }),
            ty,
            span,
        ))
    }

    fn convert_unary_expr(
        &mut self,
        unary_expr: &TypedUnaryExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let operand = self.convert_typed_expression(
            &unary_expr.operand,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;
        let op = self.convert_unary_operator(&unary_expr.operator)?;

        let ty = self.convert_evaluated_type(&unary_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Unary(TypedUnary {
                op,
                operand: Box::new(operand),
            }),
            ty,
            span,
        ))
    }

    fn convert_logic_expr(
        &mut self,
        logic_expr: &TypedLogicExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let left = self.convert_typed_expression(
            &logic_expr.left,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;
        let right = self.convert_typed_expression(
            &logic_expr.right,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        let op = match logic_expr.operator {
            LogicOperator::And | LogicOperator::AndLiteral => BinaryOp::And,
            LogicOperator::Or | LogicOperator::OrLiteral => BinaryOp::Or,
        };

        let ty = self.convert_evaluated_type(&logic_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        Ok(TypedNode::new(
            ZyntaxTypedExpression::Binary(TypedBinary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            }),
            ty,
            span,
        ))
    }

    fn convert_update_expr(
        &mut self,
        update_expr: &TypedUpdateExpr,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<ZyntaxTypedExpression>> {
        let operand = self.convert_typed_expression(
            &update_expr.operand,
            symbol_library,
            literals,
            symbol_extractor,
            type_registry,
            arena,
        )?;

        // Update operators are like ? and ! in Whirlwind
        // Map to Try/Unwrap expressions in our system
        let ty = self.convert_evaluated_type(&update_expr.inferred_type, symbol_library)?;
        let span = Span::default();

        match update_expr.operator {
            UpdateOperator::TryFrom => Ok(TypedNode::new(
                ZyntaxTypedExpression::Try(Box::new(operand)),
                ty,
                span,
            )),
            UpdateOperator::Assert => {
                // Assert (!) doesn't have a direct mapping, use Try for now
                // TODO: Add Assert variant to TypedExpression if needed
                Ok(TypedNode::new(
                    ZyntaxTypedExpression::Try(Box::new(operand)),
                    ty,
                    span,
                ))
            }
        }
    }

    pub fn convert_typed_statement(
        &mut self,
        stmt: &TypedStmnt,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        match stmt {
            TypedStmnt::ReturnStatement(ret_stmt) => {
                let value = if let Some(ref expr) = ret_stmt.value {
                    Some(Box::new(self.convert_typed_expression(
                        expr,
                        symbol_library,
                        literals,
                        symbol_extractor,
                        type_registry,
                        arena,
                    )?))
                } else {
                    None
                };

                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                let span = Span::default();

                Ok(TypedNode::new(TypedStatement::Return(value), ty, span))
            }
            TypedStmnt::ExpressionStatement(expr) | TypedStmnt::FreeExpression(expr) => {
                // Special case: if the expression is an IfExpr, convert it to an If statement
                if let TypedExpression::IfExpr(if_expr) = expr {
                    // Convert condition
                    let condition = Box::new(self.convert_typed_expression(
                        &if_expr.condition,
                        symbol_library,
                        literals,
                        symbol_extractor,
                        type_registry,
                        arena,
                    )?);

                    // Convert then block
                    let then_block = self.convert_typed_block(
                        &if_expr.consequent,
                        symbol_library,
                        literals,
                        symbol_extractor,
                        type_registry,
                        arena,
                    )?;

                    // Convert else block if present
                    let else_block = if let Some(alternate) = &if_expr.alternate {
                        // The alternate contains an expression that could be another if or a block
                        // For now, wrap it in a block
                        let else_expr = self.convert_typed_expression(
                            &alternate.expression,
                            symbol_library,
                            literals,
                            symbol_extractor,
                            type_registry,
                            arena,
                        )?;

                        Some(zyntax_typed_ast::typed_ast::TypedBlock {
                            statements: vec![TypedNode::new(
                                TypedStatement::Expression(Box::new(else_expr)),
                                Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit),
                                Span::default(),
                            )],
                            span: Span::default(),
                        })
                    } else {
                        None
                    };

                    let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                    let span = Span::default();

                    return Ok(TypedNode::new(
                        TypedStatement::If(zyntax_typed_ast::typed_ast::TypedIf {
                            condition,
                            then_block,
                            else_block,
                            span,
                        }),
                        ty,
                        span,
                    ));
                }

                // For other expressions, wrap in Expression statement
                let converted_expr = self.convert_typed_expression(
                    expr,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )?;
                let ty = converted_expr.ty.clone();
                let span = converted_expr.span;

                Ok(TypedNode::new(
                    TypedStatement::Expression(Box::new(converted_expr)),
                    ty,
                    span,
                ))
            }
            TypedStmnt::VariableDeclaration(var_decl) => {
                if var_decl.names.is_empty() {
                    return Err(AdapterError::statement_conversion(
                        "Empty variable declaration",
                    ));
                }

                let symbol = symbol_library.get(var_decl.names[0]).ok_or_else(|| {
                    AdapterError::statement_conversion("Variable symbol not found")
                })?;

                let var_name = arena.intern_string(&symbol.name);

                let initializer = if let Some(ref expr) = var_decl.value {
                    Some(Box::new(self.convert_typed_expression(
                        expr,
                        symbol_library,
                        literals,
                        symbol_extractor,
                        type_registry,
                        arena,
                    )?))
                } else {
                    None
                };

                let ty = if let Some(ref init) = initializer {
                    init.ty.clone()
                } else {
                    Type::Unknown
                };

                let span = Span::default();

                Ok(TypedNode::new(
                    TypedStatement::Let(TypedLet {
                        name: var_name,
                        ty: ty.clone(),
                        mutability: Mutability::Mutable,
                        initializer,
                        span,
                    }),
                    ty.clone(),
                    span,
                ))
            }
            TypedStmnt::ForStatement(for_stmt) => {
                // Convert the iterator expression
                let iterator = Box::new(self.convert_typed_expression(
                    &for_stmt.iterator,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )?);

                // Create a pattern for the loop variable(s)
                // For now, create a simple identifier pattern from the first item
                let pattern = if !for_stmt.items.is_empty() {
                    if let Some(item_symbol) = symbol_library.get(for_stmt.items[0]) {
                        let item_name = arena.intern_string(&item_symbol.name);
                        Box::new(TypedNode::new(
                            TypedPattern::Identifier {
                                name: item_name,
                                mutability: Mutability::Immutable,
                            },
                            Type::Unknown, // Type will be inferred from iterator
                            Span::default(),
                        ))
                    } else {
                        return Err(AdapterError::statement_conversion(
                            "For loop item symbol not found",
                        ));
                    }
                } else {
                    return Err(AdapterError::statement_conversion("For loop has no items"));
                };

                // Convert the body
                let body = self.convert_typed_block(
                    &for_stmt.body,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )?;

                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                let span = Span::default();

                Ok(TypedNode::new(
                    TypedStatement::For(TypedFor {
                        pattern,
                        iterator,
                        body,
                    }),
                    ty,
                    span,
                ))
            }
            TypedStmnt::WhileStatement(while_stmt) => {
                // Convert the condition expression
                let condition = Box::new(self.convert_typed_expression(
                    &while_stmt.condition,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )?);

                // Convert the body
                let body = self.convert_typed_block(
                    &while_stmt.body,
                    symbol_library,
                    literals,
                    symbol_extractor,
                    type_registry,
                    arena,
                )?;

                let ty = Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit);
                let span = Span::default();

                Ok(TypedNode::new(
                    TypedStatement::While(zyntax_typed_ast::typed_ast::TypedWhile {
                        condition,
                        body,
                        span,
                    }),
                    ty,
                    span,
                ))
            }
            _ => Err(AdapterError::unsupported(&format!(
                "Statement conversion not yet implemented: {:?}",
                stmt
            ))),
        }
    }

    pub fn convert_typed_block(
        &mut self,
        block: &TypedBlock,
        symbol_library: &SymbolLibrary,
        literals: &LiteralMap,
        symbol_extractor: &mut SymbolExtractor,
        type_registry: &TypeRegistry,
        arena: &mut AstArena,
    ) -> AdapterResult<ZyntaxTypedBlock> {
        let mut statements = Vec::new();

        for stmt in &block.statements {
            let converted_stmt = self.convert_typed_statement(
                stmt,
                symbol_library,
                literals,
                symbol_extractor,
                type_registry,
                arena,
            )?;
            statements.push(converted_stmt);
        }

        let span = Span::default();

        Ok(ZyntaxTypedBlock { statements, span })
    }

    fn convert_evaluated_type(
        &mut self,
        eval_type: &EvaluatedType,
        _symbol_library: &SymbolLibrary,
    ) -> AdapterResult<Type> {
        match eval_type {
            EvaluatedType::Void => Ok(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)),
            EvaluatedType::Never => Ok(Type::Primitive(zyntax_typed_ast::PrimitiveType::Unit)), // Map Never to Unit for now
            EvaluatedType::Unknown => Ok(Type::Unknown),
            EvaluatedType::Partial { types } => {
                let converted_types: Vec<_> = types
                    .iter()
                    .map(|t| self.convert_evaluated_type(t, _symbol_library))
                    .collect::<AdapterResult<_>>()?;
                Ok(Type::Union(converted_types))
            }
            // TODO: Implement full type conversion for all EvaluatedType variants
            _ => Ok(Type::Unknown),
        }
    }

    fn convert_binary_operator(&self, op: &BinOperator) -> AdapterResult<BinaryOp> {
        Ok(match op {
            BinOperator::Add => BinaryOp::Add,
            BinOperator::Subtract => BinaryOp::Sub,
            BinOperator::Multiply => BinaryOp::Mul,
            BinOperator::Divide => BinaryOp::Div,
            BinOperator::Remainder => BinaryOp::Rem,
            BinOperator::Equals => BinaryOp::Eq,
            BinOperator::NotEquals => BinaryOp::Ne,
            BinOperator::LessThan => BinaryOp::Lt,
            BinOperator::LessThanOrEquals => BinaryOp::Le,
            BinOperator::GreaterThan => BinaryOp::Gt,
            BinOperator::GreaterThanOrEquals => BinaryOp::Ge,
            BinOperator::BitAnd => BinaryOp::BitAnd,
            BinOperator::BitOr => BinaryOp::BitOr,
            BinOperator::LeftShift => BinaryOp::Shl,
            BinOperator::RightShift => BinaryOp::Shr,
            BinOperator::PowerOf => {
                // PowerOf (^) doesn't have a direct mapping
                // TODO: Add PowerOf to BinaryOp or convert to function call
                BinaryOp::Mul // Placeholder
            }
            BinOperator::Range => {
                // Range (..) doesn't have a direct mapping
                // TODO: Handle range properly
                BinaryOp::Add // Placeholder
            }
        })
    }

    fn convert_unary_operator(&self, op: &UnaryOperator) -> AdapterResult<UnaryOp> {
        Ok(match op {
            UnaryOperator::Minus => UnaryOp::Minus,
            UnaryOperator::Plus => UnaryOp::Plus,
            UnaryOperator::Negation | UnaryOperator::NegationLiteral => UnaryOp::Not,
        })
    }

    fn convert_assign_operator(&self, op: &AssignOperator) -> AdapterResult<BinaryOp> {
        Ok(match op {
            AssignOperator::Assign => BinaryOp::Assign,
            AssignOperator::PlusAssign => BinaryOp::Add,
            AssignOperator::MinusAssign => BinaryOp::Sub,
        })
    }
}

impl Default for TypedExpressionConverter {
    fn default() -> Self {
        Self::new()
    }
}
