//! Expression conversion from Whirlwind Expression to TypedExpression

use crate::error::{AdapterError, AdapterResult};
use crate::type_converter::TypeConverter;
use whirlwind_ast::Expression;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedCall, TypedFieldAccess, TypedIndex, TypedLiteral, TypedUnary,
};
use zyntax_typed_ast::{
    AstArena, BinaryOp, PrimitiveType, Span, Type, TypedExpression, TypedNode, UnaryOp,
};

/// Converts Whirlwind expressions to TypedAST expressions
pub struct ExpressionConverter {
    /// Type converter for expression types
    type_converter: TypeConverter,

    /// String arena for interning strings
    arena: AstArena,
}

impl ExpressionConverter {
    /// Create a new expression converter
    pub fn new(type_converter: TypeConverter) -> Self {
        Self {
            type_converter,
            arena: AstArena::new(),
        }
    }

    /// Convert a Whirlwind Expression to TypedExpression
    ///
    /// # Whirlwind Expression Mappings:
    ///
    /// - **Identifier** → `TypedExpression::Variable`
    /// - **StringLiteral** → `TypedExpression::Literal` (string)
    /// - **NumberLiteral** → `TypedExpression::Literal` (int/float)
    /// - **BooleanLiteral** → `TypedExpression::Literal` (bool)
    /// - **BinaryExpr** (+, -, *, /, ==, etc.) → `TypedExpression::Binary`
    /// - **UnaryExpr** (!, -) → `TypedExpression::Unary`
    /// - **CallExpr** → `TypedExpression::Call`
    /// - **FnExpr** (lambda) → `TypedExpression::Lambda`
    /// - **IfExpr** → `TypedExpression::If`
    /// - **ArrayExpr** → `TypedExpression::Array`
    /// - **AccessExpr** (a.b) → `TypedExpression::MemberAccess`
    /// - **IndexExpr** (a[b]) → `TypedExpression::Index`
    /// - **BlockExpr** → `TypedExpression::Block`
    /// - **AssignmentExpr** → `TypedExpression::Assignment`
    /// - **LogicExpr** (&&, ||) → `TypedExpression::Binary` with logical ops
    /// - **UpdateExpr** (a?, a!) → Special unary expressions
    /// - **ThisExpr** → `TypedExpression::This`
    pub fn convert_expression(
        &mut self,
        whirlwind_expr: &Expression,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        match whirlwind_expr {
            Expression::Identifier(ident) => {
                let var_name = self.arena.intern_string(&ident.name);
                let span = self.convert_span(&ident.span);
                // TODO: Lookup actual type from symbol table
                let ty = Type::Unknown;
                Ok(TypedNode::new(
                    TypedExpression::Variable(var_name),
                    ty,
                    span,
                ))
            }
            Expression::StringLiteral(s) => {
                let interned = self.arena.intern_string(&s.value);
                let span = self.convert_span(&s.span);
                let ty = Type::Primitive(PrimitiveType::String);
                Ok(TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::String(interned)),
                    ty,
                    span,
                ))
            }
            Expression::NumberLiteral(num) => self.convert_number_literal(&num.value, &num.span),
            Expression::BooleanLiteral(b) => {
                let span = self.convert_span(&b.span);
                let ty = Type::Primitive(PrimitiveType::Bool);
                Ok(TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::Bool(b.value)),
                    ty,
                    span,
                ))
            }
            Expression::BinaryExpr(bin) => self.convert_binary_expr(bin),
            Expression::UnaryExpr(un) => self.convert_unary_expr(un),
            Expression::CallExpr(call) => self.convert_call_expr(call),
            Expression::FnExpr(func) => self.convert_lambda_expr(func),
            Expression::IfExpr(if_expr) => self.convert_if_expr(if_expr),
            Expression::ArrayExpr(arr) => self.convert_array_expr(arr),
            Expression::AccessExpr(access) => self.convert_field_access(access),
            Expression::IndexExpr(index) => self.convert_index_expr(index),
            Expression::BlockExpr(_block) => {
                // TODO: Implement block expression conversion
                Err(AdapterError::unsupported(
                    "Block expression conversion not yet implemented",
                ))
            }
            Expression::AssignmentExpr(assign) => self.convert_assignment(assign),
            Expression::LogicExpr(logic) => self.convert_logic_expr(logic),
            Expression::UpdateExpr(update) => self.convert_update_expr(update),
            Expression::ThisExpr(this) => {
                let span = self.convert_span(&this.span);
                let ty = Type::SelfType;
                Ok(TypedNode::new(
                    TypedExpression::Variable(self.arena.intern_string("this")),
                    ty,
                    span,
                ))
            }
        }
    }

    /// Convert Whirlwind span to TypedAST span
    fn convert_span(&self, _whirlwind_span: &whirlwind_ast::Span) -> Span {
        // TODO: Properly convert span with file information
        Span::default()
    }

    /// Convert a number literal (handling int vs float)
    fn convert_number_literal(
        &mut self,
        value: &whirlwind_ast::Number,
        span: &whirlwind_ast::Span,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        use whirlwind_ast::Number;

        let converted_span = self.convert_span(span);

        match value {
            Number::Decimal(s) | Number::Binary(s) | Number::Octal(s) | Number::Hexadecimal(s) => {
                // Try to parse as integer first
                if let Ok(int_val) = s.parse::<i128>() {
                    Ok(TypedNode::new(
                        TypedExpression::Literal(TypedLiteral::Integer(int_val)),
                        Type::Primitive(PrimitiveType::I64), // Default to i64
                        converted_span,
                    ))
                } else if let Ok(float_val) = s.parse::<f64>() {
                    Ok(TypedNode::new(
                        TypedExpression::Literal(TypedLiteral::Float(float_val)),
                        Type::Primitive(PrimitiveType::F64),
                        converted_span,
                    ))
                } else {
                    Err(AdapterError::expression_conversion(format!(
                        "Invalid number literal: {}",
                        s
                    )))
                }
            }
            Number::None => Err(AdapterError::expression_conversion("Empty number literal")),
        }
    }

    /// Convert a binary expression (arithmetic, comparison, etc.)
    fn convert_binary_expr(
        &mut self,
        bin: &whirlwind_ast::BinaryExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let left = Box::new(self.convert_expression(&bin.left)?);
        let right = Box::new(self.convert_expression(&bin.right)?);
        let op = self.convert_binary_op(&bin.operator)?;
        let span = self.convert_span(&bin.span);

        // TODO: Proper type inference
        let ty = Type::Unknown;

        Ok(TypedNode::new(
            TypedExpression::Binary(TypedBinary { op, left, right }),
            ty,
            span,
        ))
    }

    /// Convert Whirlwind binary operator to TypedAST BinaryOp
    fn convert_binary_op(&self, op: &whirlwind_ast::BinOperator) -> AdapterResult<BinaryOp> {
        use whirlwind_ast::BinOperator;

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
                return Err(AdapterError::unsupported("PowerOf operator not supported"))
            }
            BinOperator::Range => {
                return Err(AdapterError::unsupported(
                    "Range operator needs special handling",
                ))
            }
        })
    }

    /// Convert a unary expression
    fn convert_unary_expr(
        &mut self,
        un: &whirlwind_ast::UnaryExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let operand = Box::new(self.convert_expression(&un.operand)?);
        let op = self.convert_unary_op(&un.operator)?;
        let span = self.convert_span(&un.span);

        // TODO: Proper type inference
        let ty = Type::Unknown;

        Ok(TypedNode::new(
            TypedExpression::Unary(TypedUnary { op, operand }),
            ty,
            span,
        ))
    }

    /// Convert Whirlwind unary operator to TypedAST UnaryOp
    fn convert_unary_op(&self, op: &whirlwind_ast::UnaryOperator) -> AdapterResult<UnaryOp> {
        use whirlwind_ast::UnaryOperator;

        Ok(match op {
            UnaryOperator::Plus => UnaryOp::Plus,
            UnaryOperator::Minus => UnaryOp::Minus,
            UnaryOperator::Negation | UnaryOperator::NegationLiteral => UnaryOp::Not,
        })
    }

    /// Convert a call expression
    fn convert_call_expr(
        &mut self,
        call: &whirlwind_ast::CallExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let callee = Box::new(self.convert_expression(&call.caller)?);
        let positional_args: AdapterResult<Vec<TypedNode<TypedExpression>>> = call
            .arguments
            .iter()
            .map(|arg| self.convert_expression(arg))
            .collect();
        let positional_args = positional_args?;
        let span = self.convert_span(&call.span);

        // TODO: Proper type inference
        let ty = Type::Unknown;

        Ok(TypedNode::new(
            TypedExpression::Call(TypedCall {
                callee,
                positional_args,
                named_args: vec![],
                type_args: vec![],
            }),
            ty,
            span,
        ))
    }

    /// Convert a lambda/function expression
    fn convert_lambda_expr(
        &mut self,
        _func: &whirlwind_ast::FunctionExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        // TODO: Implement lambda conversion
        Err(AdapterError::unsupported(
            "Lambda expression conversion not yet implemented",
        ))
    }

    /// Convert an if expression
    fn convert_if_expr(
        &mut self,
        _if_expr: &whirlwind_ast::IfExpression,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        // TODO: Implement if expression conversion
        Err(AdapterError::unsupported(
            "If expression conversion not yet implemented",
        ))
    }

    /// Convert an array expression
    fn convert_array_expr(
        &mut self,
        arr: &whirlwind_ast::ArrayExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let elements: AdapterResult<Vec<TypedNode<TypedExpression>>> = arr
            .elements
            .iter()
            .map(|elem| self.convert_expression(elem))
            .collect();
        let elements = elements?;
        let span = self.convert_span(&arr.span);

        // TODO: Proper type inference for array type
        let ty = Type::Unknown;

        Ok(TypedNode::new(TypedExpression::Array(elements), ty, span))
    }

    /// Convert a field access expression
    fn convert_field_access(
        &mut self,
        access: &whirlwind_ast::AccessExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let object = Box::new(self.convert_expression(&access.object)?);

        // Extract field name from property expression
        let field = if let Expression::Identifier(ident) = &access.property {
            self.arena.intern_string(&ident.name)
        } else {
            return Err(AdapterError::expression_conversion(
                "Field access property must be an identifier",
            ));
        };

        let span = self.convert_span(&access.span);

        // TODO: Proper type inference
        let ty = Type::Unknown;

        Ok(TypedNode::new(
            TypedExpression::Field(TypedFieldAccess { object, field }),
            ty,
            span,
        ))
    }

    /// Convert an index expression
    fn convert_index_expr(
        &mut self,
        index: &whirlwind_ast::IndexExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let object = Box::new(self.convert_expression(&index.object)?);
        let index_expr = Box::new(self.convert_expression(&index.index)?);
        let span = self.convert_span(&index.span);

        // TODO: Proper type inference
        let ty = Type::Unknown;

        Ok(TypedNode::new(
            TypedExpression::Index(TypedIndex {
                object,
                index: index_expr,
            }),
            ty,
            span,
        ))
    }

    /// Convert an assignment expression
    fn convert_assignment(
        &mut self,
        assign: &whirlwind_ast::AssignmentExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let left = Box::new(self.convert_expression(&assign.left)?);
        let right = Box::new(self.convert_expression(&assign.right)?);
        let span = self.convert_span(&assign.span);

        // Assignment is a binary operation with Assign op
        let ty = Type::Primitive(PrimitiveType::Unit);

        Ok(TypedNode::new(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Assign,
                left,
                right,
            }),
            ty,
            span,
        ))
    }

    /// Convert a logic expression (&&, ||)
    fn convert_logic_expr(
        &mut self,
        logic: &whirlwind_ast::LogicExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let left = Box::new(self.convert_expression(&logic.left)?);
        let right = Box::new(self.convert_expression(&logic.right)?);
        let span = self.convert_span(&logic.span);

        let op = match logic.operator {
            whirlwind_ast::LogicOperator::And | whirlwind_ast::LogicOperator::AndLiteral => {
                BinaryOp::And
            }
            whirlwind_ast::LogicOperator::Or | whirlwind_ast::LogicOperator::OrLiteral => {
                BinaryOp::Or
            }
        };

        let ty = Type::Primitive(PrimitiveType::Bool);

        Ok(TypedNode::new(
            TypedExpression::Binary(TypedBinary { op, left, right }),
            ty,
            span,
        ))
    }

    /// Convert an update expression (a?, a!)
    fn convert_update_expr(
        &mut self,
        update: &whirlwind_ast::UpdateExpr,
    ) -> AdapterResult<TypedNode<TypedExpression>> {
        let operand = self.convert_expression(&update.operand)?;
        let span = self.convert_span(&update.span);

        // Map update operators to TypedAST operations
        match update.operator {
            whirlwind_ast::UpdateOperator::TryFrom => {
                // a? -> Try(a)
                Ok(TypedNode::new(
                    TypedExpression::Try(Box::new(operand)),
                    Type::Unknown, // TODO: Proper type inference
                    span,
                ))
            }
            whirlwind_ast::UpdateOperator::Assert => {
                // a! -> unwrap/assert operation - could be modeled as a special call
                // For now, just pass through the operand
                Ok(operand)
            }
        }
    }

    /// Get a mutable reference to the type converter
    pub fn type_converter_mut(&mut self) -> &mut TypeConverter {
        &mut self.type_converter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_converter_creation() {
        let type_converter = TypeConverter::new();
        let _expr_converter = ExpressionConverter::new(type_converter);
        // Basic smoke test
    }
}
