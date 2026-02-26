//! Statement conversion from Whirlwind Statement to TypedStatement

use crate::error::{AdapterError, AdapterResult};
use crate::expression_converter::ExpressionConverter;
use crate::type_converter::TypeConverter;
use whirlwind_ast::Statement;
use zyntax_typed_ast::typed_ast::{TypedBlock, TypedLet, TypedWhile};
use zyntax_typed_ast::{
    AstArena, Mutability, PrimitiveType, Span, Type, TypedNode, TypedStatement,
};

/// Converts Whirlwind statements to TypedAST statements
pub struct StatementConverter {
    /// Expression converter for nested expressions
    expression_converter: ExpressionConverter,

    /// String arena for interning strings
    arena: AstArena,
}

impl StatementConverter {
    /// Create a new statement converter
    pub fn new(type_converter: TypeConverter) -> Self {
        Self {
            expression_converter: ExpressionConverter::new(type_converter),
            arena: AstArena::new(),
        }
    }

    /// Convert a Whirlwind Statement to TypedStatement
    ///
    /// # Whirlwind Statement Mappings:
    ///
    /// **Declarations:**
    /// - **Function** → `TypedStatement::Function`
    /// - **Variable** → `TypedStatement::VariableDeclaration`
    /// - **Model** (class) → `TypedStatement::Class`
    /// - **Interface** → `TypedStatement::Interface`
    /// - **Enum** → `TypedStatement::Enum`
    /// - **Record** → `TypedStatement::Struct`
    /// - **Type** (type alias) → `TypedStatement::TypeAlias`
    ///
    /// **Control Flow:**
    /// - **While** → `TypedStatement::While`
    /// - **For** → `TypedStatement::For`
    /// - **Return** → `TypedStatement::Return`
    /// - **Continue** → `TypedStatement::Continue`
    /// - **Break** → `TypedStatement::Break`
    ///
    /// **Expressions:**
    /// - **Expression** (with semicolon) → `TypedStatement::Expression`
    /// - **FreeExpression** (without semicolon) → Last statement in block
    ///
    /// **Imports:**
    /// - **Use** → `TypedStatement::Import`
    /// - **Import** → `TypedStatement::Import`
    ///
    /// **Testing:**
    /// - **Test** → `TypedStatement::Test` (or custom attribute)
    pub fn convert_statement(
        &mut self,
        whirlwind_stmt: &Statement,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        match whirlwind_stmt {
            Statement::FunctionDeclaration(_func_decl) => {
                // TODO: Implement function declaration
                // This requires converting the function signature, parameters, and body
                Err(AdapterError::unsupported(
                    "Function declaration not yet implemented",
                ))
            }
            Statement::VariableDeclaration(var_decl) => self.convert_variable_declaration(var_decl),
            Statement::ShorthandVariableDeclaration(short_decl) => {
                self.convert_shorthand_variable(short_decl)
            }
            Statement::ModelDeclaration(_model) => Err(AdapterError::unsupported(
                "Model/Class declaration not yet implemented",
            )),
            Statement::InterfaceDeclaration(_interface) => Err(AdapterError::unsupported(
                "Interface declaration not yet implemented",
            )),
            Statement::EnumDeclaration(_enum_decl) => Err(AdapterError::unsupported(
                "Enum declaration not yet implemented",
            )),
            Statement::RecordDeclaration => Err(AdapterError::unsupported(
                "Record declaration not yet implemented",
            )),
            Statement::TypeEquation(_type_eq) => {
                Err(AdapterError::unsupported("Type alias not yet implemented"))
            }
            Statement::WhileStatement(while_stmt) => self.convert_while_statement(while_stmt),
            Statement::ForStatement(_for_stmt) => {
                Err(AdapterError::unsupported("For loop not yet implemented"))
            }
            Statement::ReturnStatement(ret_stmt) => self.convert_return_statement(ret_stmt),
            Statement::ContinueStatement(_cont) => {
                let span = Span::default(); // TODO: Get actual span
                Ok(TypedNode::new(
                    TypedStatement::Continue,
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ))
            }
            Statement::BreakStatement(_brk) => {
                let span = Span::default(); // TODO: Get actual span
                Ok(TypedNode::new(
                    TypedStatement::Break(None),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ))
            }
            Statement::ExpressionStatement(expr) => {
                let typed_expr = self.expression_converter.convert_expression(expr)?;
                let ty = Type::Primitive(PrimitiveType::Unit);
                let span = typed_expr.span;
                Ok(TypedNode::new(
                    TypedStatement::Expression(Box::new(typed_expr)),
                    ty,
                    span,
                ))
            }
            Statement::FreeExpression(expr) => {
                // Free expression (without semicolon) - treat as expression statement
                let typed_expr = self.expression_converter.convert_expression(expr)?;
                let ty = typed_expr.ty.clone();
                let span = typed_expr.span;
                Ok(TypedNode::new(
                    TypedStatement::Expression(Box::new(typed_expr)),
                    ty,
                    span,
                ))
            }
            Statement::UseDeclaration(_use_decl) => Err(AdapterError::unsupported(
                "Use declaration not yet implemented",
            )),
            Statement::ImportDeclaration(_import) => Err(AdapterError::unsupported(
                "Import declaration not yet implemented",
            )),
            Statement::TestDeclaration(_test) => Err(AdapterError::unsupported(
                "Test declaration not yet implemented",
            )),
            Statement::ModuleDeclaration(_module) => Err(AdapterError::unsupported(
                "Module declaration not yet implemented",
            )),
        }
    }

    /// Convert a variable declaration
    fn convert_variable_declaration(
        &mut self,
        var_decl: &whirlwind_ast::VariableDeclaration,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        // Whirlwind allows multiple variables in one declaration
        // For now, just handle the first one
        if var_decl.addresses.is_empty() {
            return Err(AdapterError::statement_conversion(
                "Empty variable declaration",
            ));
        }

        let first_addr = &var_decl.addresses[0];
        // TODO: Look up actual name from symbol table using scope address
        let var_name = self
            .arena
            .intern_string(&format!("var_{}", first_addr.entry_no));

        let initializer = if let Some(ref expr) = var_decl.value {
            Some(Box::new(
                self.expression_converter.convert_expression(expr)?,
            ))
        } else {
            None
        };

        // TODO: Get actual type from type annotation in scope address
        let ty = if let Some(ref init) = initializer {
            init.ty.clone()
        } else {
            Type::Unknown
        };

        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: var_name,
                ty: ty.clone(),
                mutability: Mutability::Mutable, // TODO: Determine from declaration
                initializer,
                span,
            }),
            ty,
            span,
        ))
    }

    /// Convert a shorthand variable declaration (`:=`)
    fn convert_shorthand_variable(
        &mut self,
        short_decl: &whirlwind_ast::ShorthandVariableDeclaration,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        // TODO: Look up actual name from symbol table using scope address
        let var_name = self
            .arena
            .intern_string(&format!("var_{}", short_decl.address.entry_no));
        let initializer = self
            .expression_converter
            .convert_expression(&short_decl.value)?;
        let ty = initializer.ty.clone();
        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: var_name,
                ty: ty.clone(),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(initializer)),
                span,
            }),
            ty,
            span,
        ))
    }

    /// Convert a while statement
    fn convert_while_statement(
        &mut self,
        while_stmt: &whirlwind_ast::WhileStatement,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        let condition = Box::new(
            self.expression_converter
                .convert_expression(&while_stmt.condition)?,
        );
        let body = self.convert_block(&while_stmt.body)?;
        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedStatement::While(TypedWhile {
                condition,
                body,
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a return statement
    fn convert_return_statement(
        &mut self,
        ret_stmt: &whirlwind_ast::ReturnStatement,
    ) -> AdapterResult<TypedNode<TypedStatement>> {
        let value = if let Some(ref expr) = ret_stmt.value {
            let typed_expr = self.expression_converter.convert_expression(expr)?;
            Some(Box::new(typed_expr))
        } else {
            None
        };

        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedNode::new(
            TypedStatement::Return(value),
            Type::Primitive(PrimitiveType::Unit),
            span,
        ))
    }

    /// Convert a Whirlwind Block to TypedBlock
    fn convert_block(&mut self, block: &whirlwind_ast::Block) -> AdapterResult<TypedBlock> {
        let statements: AdapterResult<Vec<TypedNode<TypedStatement>>> = block
            .statements
            .iter()
            .map(|stmt| self.convert_statement(stmt))
            .collect();

        let statements = statements?;
        let span = Span::default(); // TODO: Convert actual span

        Ok(TypedBlock { statements, span })
    }

    /// Get a mutable reference to the expression converter
    pub fn expression_converter_mut(&mut self) -> &mut ExpressionConverter {
        &mut self.expression_converter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_converter_creation() {
        let type_converter = TypeConverter::new();
        let _stmt_converter = StatementConverter::new(type_converter);
        // Basic smoke test
    }
}
