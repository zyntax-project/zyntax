//! # Comprehensive TypedAST Builder
//!
//! Fluent builder API for constructing well-formed TypedAST nodes.
//! Provides type-safe construction methods for all TypedAST patterns including:
//! - Enhanced parameter system with named arguments
//! - Pattern matching constructs
//! - Coroutine and async statements
//! - All expression and statement types

use crate::arena::{AstArena, InternedString};
use crate::source::Span;
use crate::type_registry::{
    CallingConvention, Mutability, ParamInfo, PrimitiveType, Type, Visibility,
};
use crate::typed_ast::*;

use crate::type_registry::TypeRegistry;

/// Comprehensive fluent builder for TypedAST
pub struct TypedASTBuilder {
    arena: AstArena,
    pub registry: TypeRegistry,
    source_file: Option<String>,
    source_content: Option<String>,
}

impl TypedASTBuilder {
    pub fn new() -> Self {
        Self {
            arena: AstArena::new(),
            registry: TypeRegistry::new(),
            source_file: None,
            source_content: None,
        }
    }

    /// Set the source file information for span tracking
    pub fn set_source(&mut self, file_name: String, content: String) {
        self.source_file = Some(file_name);
        self.source_content = Some(content);
    }

    /// Get the source file name
    pub fn source_file(&self) -> Option<&str> {
        self.source_file.as_deref()
    }

    /// Get the source content
    pub fn source_content(&self) -> Option<&str> {
        self.source_content.as_deref()
    }

    /// Get a reference to the arena for string interning
    pub fn arena(&mut self) -> &mut AstArena {
        &mut self.arena
    }

    /// Intern a string
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.arena.intern_string(s)
    }

    // ====== SPAN HELPERS ======

    /// Create a span from start and end positions
    pub fn span(&self, start: usize, end: usize) -> Span {
        Span::new(start, end)
    }

    /// Create a dummy span for testing
    pub fn dummy_span(&self) -> Span {
        Span::new(0, 0)
    }

    // ====== TYPE HELPERS ======

    /// Get i32 primitive type
    pub fn i32_type(&self) -> Type {
        Type::Primitive(PrimitiveType::I32)
    }

    /// Get i64 primitive type
    pub fn i64_type(&self) -> Type {
        Type::Primitive(PrimitiveType::I64)
    }

    /// Get bool primitive type
    pub fn bool_type(&self) -> Type {
        Type::Primitive(PrimitiveType::Bool)
    }

    /// Get string primitive type
    pub fn string_type(&self) -> Type {
        Type::Primitive(PrimitiveType::String)
    }

    /// Get unit primitive type
    pub fn unit_type(&self) -> Type {
        Type::Primitive(PrimitiveType::Unit)
    }

    /// Get char primitive type
    pub fn char_type(&self) -> Type {
        Type::Primitive(PrimitiveType::Char)
    }

    /// Get f32 primitive type
    pub fn f32_type(&self) -> Type {
        Type::Primitive(PrimitiveType::F32)
    }

    /// Get f64 primitive type
    pub fn f64_type(&self) -> Type {
        Type::Primitive(PrimitiveType::F64)
    }

    // ====== EXPRESSION BUILDERS ======

    /// Build integer literal
    pub fn int_literal(&mut self, value: i128, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Integer(value)),
            Type::Primitive(PrimitiveType::I32),
            span,
        )
    }

    /// Build float literal (defaults to F32 for ML-focused usage)
    pub fn float_literal(&mut self, value: f64, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Float(value)),
            Type::Primitive(PrimitiveType::F32),
            span,
        )
    }

    /// Build F64 float literal
    pub fn float64_literal(&mut self, value: f64, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Float(value)),
            Type::Primitive(PrimitiveType::F64),
            span,
        )
    }

    /// Build string literal
    pub fn string_literal(&mut self, value: &str, span: Span) -> TypedNode<TypedExpression> {
        let interned = self.intern(value);
        typed_node(
            TypedExpression::Literal(TypedLiteral::String(interned)),
            Type::Primitive(PrimitiveType::String),
            span,
        )
    }

    /// Build boolean literal
    pub fn bool_literal(&mut self, value: bool, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Bool(value)),
            Type::Primitive(PrimitiveType::Bool),
            span,
        )
    }

    /// Build character literal
    pub fn char_literal(&mut self, value: char, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Char(value)),
            Type::Primitive(PrimitiveType::Char),
            span,
        )
    }

    /// Build unit literal
    pub fn unit_literal(&mut self, span: Span) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Literal(TypedLiteral::Unit),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build variable reference
    pub fn variable(&mut self, name: &str, ty: Type, span: Span) -> TypedNode<TypedExpression> {
        let interned = self.intern(name);
        typed_node(TypedExpression::Variable(interned), ty, span)
    }

    /// Build binary expression
    pub fn binary(
        &mut self,
        op: BinaryOp,
        left: TypedNode<TypedExpression>,
        right: TypedNode<TypedExpression>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Binary(TypedBinary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }),
            result_type,
            span,
        )
    }

    /// Build unary expression
    pub fn unary(
        &mut self,
        op: UnaryOp,
        operand: TypedNode<TypedExpression>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Unary(TypedUnary {
                op,
                operand: Box::new(operand),
            }),
            result_type,
            span,
        )
    }

    /// Build if expression (ternary conditional)
    pub fn if_expr(
        &mut self,
        condition: TypedNode<TypedExpression>,
        then_branch: TypedNode<TypedExpression>,
        else_branch: TypedNode<TypedExpression>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::If(TypedIfExpr {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            }),
            result_type,
            span,
        )
    }

    /// Build function call with only positional arguments
    pub fn call_positional(
        &mut self,
        callee: TypedNode<TypedExpression>,
        args: Vec<TypedNode<TypedExpression>>,
        return_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Call(TypedCall::positional(callee, args)),
            return_type,
            span,
        )
    }

    /// Build function call with named arguments
    pub fn call_named(
        &mut self,
        callee: TypedNode<TypedExpression>,
        args: Vec<(&str, TypedNode<TypedExpression>)>,
        return_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let named_args = args
            .into_iter()
            .map(|(name, expr)| TypedNamedArg::new(self.intern(name), expr, span))
            .collect();

        typed_node(
            TypedExpression::Call(TypedCall::named_only(callee, named_args)),
            return_type,
            span,
        )
    }

    /// Build function call with mixed positional and named arguments
    pub fn call_mixed(
        &mut self,
        callee: TypedNode<TypedExpression>,
        positional: Vec<TypedNode<TypedExpression>>,
        named: Vec<(&str, TypedNode<TypedExpression>)>,
        type_args: Vec<Type>,
        return_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let named_args = named
            .into_iter()
            .map(|(name, expr)| TypedNamedArg::new(self.intern(name), expr, span))
            .collect();

        typed_node(
            TypedExpression::Call(TypedCall::mixed(callee, positional, named_args, type_args)),
            return_type,
            span,
        )
    }

    /// Build method call
    pub fn method_call(
        &mut self,
        receiver: TypedNode<TypedExpression>,
        method: &str,
        args: Vec<TypedNode<TypedExpression>>,
        return_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let method_name = self.intern(method);
        typed_node(
            TypedExpression::MethodCall(TypedMethodCall::positional(receiver, method_name, args)),
            return_type,
            span,
        )
    }

    /// Build field access
    pub fn field_access(
        &mut self,
        object: TypedNode<TypedExpression>,
        field: &str,
        field_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let field_name = self.intern(field);
        typed_node(
            TypedExpression::Field(TypedFieldAccess {
                object: Box::new(object),
                field: field_name,
            }),
            field_type,
            span,
        )
    }

    /// Build array/slice indexing
    pub fn index(
        &mut self,
        object: TypedNode<TypedExpression>,
        index: TypedNode<TypedExpression>,
        element_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Index(TypedIndex {
                object: Box::new(object),
                index: Box::new(index),
            }),
            element_type,
            span,
        )
    }

    /// Build struct literal
    pub fn struct_literal(
        &mut self,
        name: &str,
        fields: Vec<(&str, TypedNode<TypedExpression>)>,
        struct_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let struct_name = self.intern(name);
        let field_inits = fields
            .into_iter()
            .map(|(name, expr)| TypedFieldInit {
                name: self.intern(name),
                value: Box::new(expr),
            })
            .collect();

        typed_node(
            TypedExpression::Struct(TypedStructLiteral {
                name: struct_name,
                fields: field_inits,
            }),
            struct_type,
            span,
        )
    }

    /// Build lambda expression
    pub fn lambda(
        &mut self,
        params: Vec<(&str, Option<Type>)>,
        body: TypedNode<TypedExpression>,
        lambda_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        let lambda_params = params
            .into_iter()
            .map(|(name, ty)| TypedLambdaParam {
                name: self.intern(name),
                ty,
            })
            .collect();

        typed_node(
            TypedExpression::Lambda(TypedLambda {
                params: lambda_params,
                body: TypedLambdaBody::Expression(Box::new(body)),
                captures: vec![],
            }),
            lambda_type,
            span,
        )
    }

    /// Build cast expression
    pub fn cast(
        &mut self,
        expr: TypedNode<TypedExpression>,
        target_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(expr),
                target_type: target_type.clone(),
            }),
            target_type,
            span,
        )
    }

    /// Build try expression (for error handling)
    pub fn try_expr(
        &mut self,
        expr: TypedNode<TypedExpression>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(TypedExpression::Try(Box::new(expr)), result_type, span)
    }

    /// Build await expression (for async)
    pub fn await_expr(
        &mut self,
        expr: TypedNode<TypedExpression>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(TypedExpression::Await(Box::new(expr)), result_type, span)
    }

    /// Build reference expression (&x)
    pub fn reference(
        &mut self,
        expr: TypedNode<TypedExpression>,
        mutability: Mutability,
        pointer_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Reference(TypedReference {
                expr: Box::new(expr),
                mutability,
            }),
            pointer_type,
            span,
        )
    }

    /// Build dereference expression (*x)
    pub fn dereference(
        &mut self,
        expr: TypedNode<TypedExpression>,
        deref_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Dereference(Box::new(expr)),
            deref_type,
            span,
        )
    }

    /// Build range expression (start..end)
    pub fn range(
        &mut self,
        start: Option<TypedNode<TypedExpression>>,
        end: Option<TypedNode<TypedExpression>>,
        inclusive: bool,
        range_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(
            TypedExpression::Range(TypedRange {
                start: start.map(Box::new),
                end: end.map(Box::new),
                inclusive,
            }),
            range_type,
            span,
        )
    }

    /// Build tuple expression
    pub fn tuple(
        &mut self,
        elements: Vec<TypedNode<TypedExpression>>,
        tuple_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(TypedExpression::Tuple(elements), tuple_type, span)
    }

    /// Build array literal expression
    pub fn array_literal(
        &mut self,
        elements: Vec<TypedNode<TypedExpression>>,
        array_type: Type,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        typed_node(TypedExpression::Array(elements), array_type, span)
    }

    // ====== STATEMENT BUILDERS ======

    /// Build let statement
    pub fn let_statement(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        initializer: Option<TypedNode<TypedExpression>>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        let var_name = self.intern(name);
        typed_node(
            TypedStatement::Let(TypedLet {
                name: var_name,
                ty: ty.clone(),
                mutability,
                initializer: initializer.map(Box::new),
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build if statement
    pub fn if_statement(
        &mut self,
        condition: TypedNode<TypedExpression>,
        then_block: TypedBlock,
        else_block: Option<TypedBlock>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::If(TypedIf {
                condition: Box::new(condition),
                then_block,
                else_block,
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build while loop
    pub fn while_loop(
        &mut self,
        condition: TypedNode<TypedExpression>,
        body: TypedBlock,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::While(TypedWhile {
                condition: Box::new(condition),
                body,
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build for loop (foreach style)
    pub fn for_loop(
        &mut self,
        variable: &str,
        iterable: TypedNode<TypedExpression>,
        body: TypedBlock,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        let var_name = self.intern(variable);
        typed_node(
            TypedStatement::For(TypedFor {
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(var_name),
                    Type::Never,
                    span,
                )),
                iterator: Box::new(iterable),
                body,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build match statement
    pub fn match_statement(
        &mut self,
        scrutinee: TypedNode<TypedExpression>,
        arms: Vec<(
            TypedNode<TypedPattern>,
            Option<TypedNode<TypedExpression>>,
            TypedNode<TypedExpression>,
        )>,
        result_type: Type,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        let match_arms = arms
            .into_iter()
            .map(|(pattern, guard, body)| TypedMatchArm {
                pattern: Box::new(pattern),
                guard: guard.map(Box::new),
                body: Box::new(body),
            })
            .collect();

        typed_node(
            TypedStatement::Match(TypedMatch {
                scrutinee: Box::new(scrutinee),
                arms: match_arms,
            }),
            result_type,
            span,
        )
    }

    /// Build expression statement
    pub fn expression_statement(
        &mut self,
        expr: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Expression(Box::new(expr)),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build block
    pub fn block(&mut self, statements: Vec<TypedNode<TypedStatement>>, span: Span) -> TypedBlock {
        TypedBlock { statements, span }
    }

    /// Build return statement
    pub fn return_stmt(
        &mut self,
        value: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Return(Some(Box::new(value))),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build return statement with no value (void return)
    pub fn return_void(&mut self, span: Span) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Return(None),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build break statement
    pub fn break_stmt(&mut self, span: Span) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Break(None),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build break statement with value
    pub fn break_with_value(
        &mut self,
        value: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Break(Some(Box::new(value))),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build continue statement
    pub fn continue_stmt(&mut self, span: Span) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Continue,
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build throw statement
    pub fn throw_stmt(
        &mut self,
        exception: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Throw(Box::new(exception)),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build loop statement (infinite loop)
    pub fn loop_stmt(&mut self, body: TypedBlock, span: Span) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Loop(TypedLoop::Infinite { body }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    // ====== PATTERN BUILDERS ======

    /// Build struct pattern
    pub fn struct_pattern(
        &mut self,
        name: &str,
        fields: Vec<(&str, TypedNode<TypedPattern>)>,
        span: Span,
    ) -> TypedNode<TypedPattern> {
        let struct_name = self.intern(name);
        let field_patterns = fields
            .into_iter()
            .map(|(name, pattern)| TypedFieldPattern {
                name: self.intern(name),
                pattern: Box::new(pattern),
            })
            .collect();

        typed_node(
            TypedPattern::Struct {
                name: struct_name,
                fields: field_patterns,
            },
            Type::Never,
            span,
        )
    }

    /// Build enum pattern
    pub fn enum_pattern(
        &mut self,
        enum_name: &str,
        variant: &str,
        fields: Vec<TypedNode<TypedPattern>>,
        span: Span,
    ) -> TypedNode<TypedPattern> {
        let enum_interned = self.intern(enum_name);
        let variant_interned = self.intern(variant);

        typed_node(
            TypedPattern::Enum {
                name: enum_interned,
                variant: variant_interned,
                fields,
            },
            Type::Never,
            span,
        )
    }

    /// Build array pattern
    pub fn array_pattern(
        &mut self,
        patterns: Vec<TypedNode<TypedPattern>>,
        span: Span,
    ) -> TypedNode<TypedPattern> {
        typed_node(TypedPattern::Array(patterns), Type::Never, span)
    }

    /// Build slice pattern with prefix, middle (rest), and suffix
    pub fn slice_pattern(
        &mut self,
        prefix: Vec<TypedNode<TypedPattern>>,
        middle: Option<TypedNode<TypedPattern>>,
        suffix: Vec<TypedNode<TypedPattern>>,
        span: Span,
    ) -> TypedNode<TypedPattern> {
        typed_node(
            TypedPattern::Slice {
                prefix,
                middle: middle.map(Box::new),
                suffix,
            },
            Type::Never,
            span,
        )
    }

    /// Build map pattern
    pub fn map_pattern(
        &mut self,
        entries: Vec<(&str, TypedNode<TypedPattern>)>,
        rest: Option<(&str, Mutability)>,
        exhaustive: bool,
        span: Span,
    ) -> TypedNode<TypedPattern> {
        let mut pattern_entries = Vec::new();

        // Add key-value entries
        for (key, pattern) in entries {
            pattern_entries.push(TypedMapPatternEntry::KeyValue {
                key: typed_node(
                    TypedLiteralPattern::String(self.intern(key)),
                    Type::Primitive(PrimitiveType::String),
                    span,
                ),
                pattern: Box::new(pattern),
            });
        }

        // Add rest pattern if specified
        if let Some((rest_name, mutability)) = rest {
            pattern_entries.push(TypedMapPatternEntry::Rest {
                name: Some(self.intern(rest_name)),
                mutability,
            });
        }

        typed_node(
            TypedPattern::Map(TypedMapPattern {
                entries: pattern_entries,
                exhaustive,
            }),
            Type::Never,
            span,
        )
    }

    // ====== PARAMETER BUILDERS ======

    /// Build regular parameter
    pub fn parameter(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        span: Span,
    ) -> TypedParameter {
        TypedParameter::regular(self.intern(name), ty, mutability, span)
    }

    /// Build optional parameter with default value
    pub fn optional_parameter(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        default: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedParameter {
        TypedParameter::optional(self.intern(name), ty, mutability, default, span)
    }

    /// Build rest/variadic parameter
    pub fn rest_parameter(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        span: Span,
    ) -> TypedParameter {
        TypedParameter::rest(self.intern(name), ty, mutability, span)
    }

    /// Build out parameter (C#-style)
    pub fn out_parameter(&mut self, name: &str, ty: Type, span: Span) -> TypedParameter {
        TypedParameter::out(self.intern(name), ty, span)
    }

    /// Build ref parameter (C#-style)
    pub fn ref_parameter(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        span: Span,
    ) -> TypedParameter {
        TypedParameter::ref_param(self.intern(name), ty, mutability, span)
    }

    /// Build inout parameter (Swift-style)
    pub fn inout_parameter(&mut self, name: &str, ty: Type, span: Span) -> TypedParameter {
        TypedParameter::inout(self.intern(name), ty, span)
    }

    // ====== DECLARATION BUILDERS ======

    /// Build function declaration
    pub fn function(
        &mut self,
        name: &str,
        params: Vec<TypedParameter>,
        return_type: Type,
        body: TypedBlock,
        visibility: Visibility,
        is_async: bool,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        let func_name = self.intern(name);
        typed_node(
            TypedDeclaration::Function(TypedFunction {
                name: func_name,
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params,
                return_type: return_type.clone(),
                body: Some(body),
                visibility,
                is_async,
                is_pure: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
            }),
            return_type,
            span,
        )
    }

    /// Build extern function declaration (no body)
    pub fn extern_function(
        &mut self,
        name: &str,
        params: Vec<TypedParameter>,
        return_type: Type,
        visibility: Visibility,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        let func_name = self.intern(name);
        typed_node(
            TypedDeclaration::Function(TypedFunction {
                name: func_name,
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params,
                return_type: return_type.clone(),
                body: None,
                visibility,
                is_async: false,
                is_pure: false,
                is_external: true,
                calling_convention: CallingConvention::Cdecl,
                link_name: None,
            }),
            return_type,
            span,
        )
    }

    /// Build variable declaration
    pub fn variable_declaration(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        initializer: Option<TypedNode<TypedExpression>>,
        visibility: Visibility,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        let var_name = self.intern(name);
        typed_node(
            TypedDeclaration::Variable(TypedVariable {
                name: var_name,
                ty: ty.clone(),
                mutability,
                initializer: initializer.map(Box::new),
                visibility,
            }),
            ty,
            span,
        )
    }

    /// Build import declaration for a single module
    ///
    /// # Arguments
    /// * `module_name` - The module name to import (e.g., "prelude", "tensor")
    /// * `span` - Source span for error reporting
    ///
    /// # Example
    /// ```ignore
    /// // import prelude
    /// builder.import("prelude", span)
    /// ```
    pub fn import(&mut self, module_name: &str, span: Span) -> TypedNode<TypedDeclaration> {
        use crate::typed_ast::{TypedImport, TypedImportItem};

        let module_path = vec![self.intern(module_name)];

        // For simple imports like "import prelude", we treat it as a glob import
        // This makes all symbols from the module available
        let items = vec![TypedImportItem::Glob];

        typed_node(
            TypedDeclaration::Import(TypedImport {
                module_path,
                items,
                span,
            }),
            Type::Never, // Imports don't have a type
            span,
        )
    }

    /// Build trait declaration
    ///
    /// # Arguments
    /// * `name` - The trait name (e.g., "Display", "Add")
    /// * `type_params` - Type parameters for generic traits (e.g., <Rhs> in Add<Rhs>)
    /// * `methods` - Trait method signatures
    /// * `associated_types` - Associated types (e.g., type Output)
    /// * `span` - Source span for error reporting
    ///
    /// # Example
    /// ```ignore
    /// // trait Display { fn to_string(self) -> String }
    /// builder.trait_def("Display", vec![], methods, vec![], span)
    /// ```
    pub fn trait_def(
        &mut self,
        name: &str,
        type_params: Vec<TypedTypeParam>,
        methods: Vec<TypedMethodSignature>,
        associated_types: Vec<TypedAssociatedType>,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        use crate::typed_ast::TypedInterface;

        let trait_name = self.intern(name);

        typed_node(
            TypedDeclaration::Interface(TypedInterface {
                name: trait_name,
                type_params,
                extends: vec![],
                methods,
                associated_types,
                visibility: Visibility::Public,
                span,
            }),
            Type::Never, // Traits don't have a value type
            span,
        )
    }

    /// Build trait implementation block
    ///
    /// # Arguments
    /// * `trait_name` - The trait being implemented (e.g., "Add")
    /// * `trait_type_args` - Type arguments for the trait (e.g., <Tensor> in Add<Tensor>)
    /// * `for_type` - The type implementing the trait (e.g., Tensor)
    /// * `methods` - Method implementations
    /// * `associated_types` - Associated type definitions
    /// * `span` - Source span for error reporting
    ///
    /// # Example
    /// ```ignore
    /// // impl Add<Tensor> for Tensor { ... }
    /// builder.impl_block("Add", vec![tensor_type], tensor_type, methods, assoc_types, span)
    /// ```
    pub fn impl_block(
        &mut self,
        trait_name: &str,
        trait_type_args: Vec<Type>,
        for_type: Type,
        methods: Vec<TypedMethod>,
        associated_types: Vec<TypedImplAssociatedType>,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        use crate::typed_ast::TypedTraitImpl;

        let trait_name_interned = self.intern(trait_name);

        typed_node(
            TypedDeclaration::Impl(TypedTraitImpl {
                trait_name: trait_name_interned,
                trait_type_args,
                for_type: for_type.clone(),
                methods,
                associated_types,
                span,
            }),
            Type::Never, // Impls don't have a value type
            span,
        )
    }

    // ====== COROUTINE AND ASYNC BUILDERS ======

    /// Build coroutine statement
    pub fn coroutine(
        &mut self,
        kind: CoroutineKind,
        body: TypedNode<TypedExpression>,
        params: Vec<TypedNode<TypedExpression>>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Coroutine(TypedCoroutine {
                kind,
                body: Box::new(body),
                params,
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build defer statement
    pub fn defer(
        &mut self,
        body: TypedNode<TypedExpression>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        typed_node(
            TypedStatement::Defer(TypedDefer {
                body: Box::new(body),
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    /// Build select statement
    pub fn select(
        &mut self,
        arms: Vec<(TypedSelectOperation, TypedBlock)>,
        default: Option<TypedBlock>,
        span: Span,
    ) -> TypedNode<TypedStatement> {
        let select_arms = arms
            .into_iter()
            .map(|(operation, body)| TypedSelectArm {
                operation,
                body,
                span,
            })
            .collect();

        typed_node(
            TypedStatement::Select(TypedSelect {
                arms: select_arms,
                default,
                span,
            }),
            Type::Primitive(PrimitiveType::Unit),
            span,
        )
    }

    // ====== HELPER METHODS ======

    /// Create a typed program from declarations
    pub fn program(
        &mut self,
        declarations: Vec<TypedNode<TypedDeclaration>>,
        span: Span,
    ) -> TypedProgram {
        use crate::source::SourceFile;

        // Create SourceFile if we have source information
        let source_files =
            if let (Some(name), Some(content)) = (self.source_file(), self.source_content()) {
                vec![SourceFile::new(name.to_string(), content.to_string())]
            } else {
                vec![]
            };

        TypedProgram {
            declarations,
            span,
            source_files,
            type_registry: self.registry.clone(),
        }
    }
}

/// Fluent function builder for more ergonomic test construction
pub struct FluentFunctionBuilder {
    builder: TypedASTBuilder,
    name: InternedString,
    params: Vec<TypedParameter>,
    return_type: Type,
    statements: Vec<TypedNode<TypedStatement>>,
    visibility: Visibility,
    is_async: bool,
    span: Span,
}

impl FluentFunctionBuilder {
    /// Create a new fluent function builder
    pub fn new(builder: TypedASTBuilder, name: &str, span: Span) -> Self {
        let mut inner_builder = builder;
        let interned_name = inner_builder.intern(name);

        Self {
            builder: inner_builder,
            name: interned_name,
            params: Vec::new(),
            return_type: Type::Primitive(PrimitiveType::Unit),
            statements: Vec::new(),
            visibility: Visibility::Public,
            is_async: false,
            span,
        }
    }

    /// Add a regular parameter
    pub fn param(mut self, name: &str, ty: impl Into<Type>) -> Self {
        let param = self
            .builder
            .parameter(name, ty.into(), Mutability::Immutable, self.span);
        self.params.push(param);
        self
    }

    /// Add a mutable parameter
    pub fn mut_param(mut self, name: &str, ty: impl Into<Type>) -> Self {
        let param = self
            .builder
            .parameter(name, ty.into(), Mutability::Mutable, self.span);
        self.params.push(param);
        self
    }

    /// Add an optional parameter with default value
    pub fn optional_param<T>(mut self, name: &str, ty: impl Into<Type>, default: T) -> Self
    where
        T: Into<DefaultValue>,
    {
        let default_expr = default.into().to_expression(&mut self.builder, self.span);
        let param = self.builder.optional_parameter(
            name,
            ty.into(),
            Mutability::Immutable,
            default_expr,
            self.span,
        );
        self.params.push(param);
        self
    }

    /// Add an out parameter
    pub fn out_param(mut self, name: &str, ty: impl Into<Type>) -> Self {
        let param = self.builder.out_parameter(name, ty.into(), self.span);
        self.params.push(param);
        self
    }

    /// Add a rest/varargs parameter
    pub fn rest_param(mut self, name: &str, ty: impl Into<Type>) -> Self {
        let param = self
            .builder
            .rest_parameter(name, ty.into(), Mutability::Immutable, self.span);
        self.params.push(param);
        self
    }

    /// Set the return type
    pub fn returns(mut self, ty: impl Into<Type>) -> Self {
        self.return_type = ty.into();
        self
    }

    /// Add a statement to the function body
    pub fn stmt(mut self, stmt: TypedNode<TypedStatement>) -> Self {
        self.statements.push(stmt);
        self
    }

    /// Set function as async
    pub fn is_async(mut self) -> Self {
        self.is_async = true;
        self
    }

    /// Set visibility
    pub fn visibility(mut self, vis: Visibility) -> Self {
        self.visibility = vis;
        self
    }

    /// Build the function declaration
    pub fn build(self) -> TypedNode<TypedDeclaration> {
        let body = TypedBlock {
            statements: self.statements,
            span: self.span,
        };

        typed_node(
            TypedDeclaration::Function(TypedFunction {
                name: self.name,
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params: self.params,
                return_type: self.return_type.clone(),
                body: Some(body),
                visibility: self.visibility,
                is_async: self.is_async,
                is_pure: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
            }),
            Type::Function {
                params: vec![], // Would need proper conversion from params
                return_type: Box::new(self.return_type),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: crate::type_registry::AsyncKind::default(),
                calling_convention: crate::type_registry::CallingConvention::default(),
                nullability: crate::type_registry::NullabilityKind::default(),
            },
            self.span,
        )
    }

    /// Build function and also return the function type for variable declarations
    pub fn build_with_type(self) -> (TypedNode<TypedDeclaration>, Type) {
        // Convert parameters to ParamInfo for function type
        let param_infos: Vec<ParamInfo> = self
            .params
            .iter()
            .map(|p| ParamInfo {
                name: Some(p.name),
                ty: p.ty.clone(),
                is_optional: matches!(p.kind, ParameterKind::Optional),
                is_varargs: matches!(p.kind, ParameterKind::Rest),
                is_keyword_only: false,
                is_positional_only: false,
                is_out: matches!(p.kind, ParameterKind::Out),
                is_ref: matches!(p.kind, ParameterKind::Ref),
                is_inout: matches!(p.kind, ParameterKind::InOut),
            })
            .collect();

        let has_named_params = true; // For flexibility
        let has_default_params = self
            .params
            .iter()
            .any(|p| matches!(p.kind, ParameterKind::Optional));
        let is_varargs = self
            .params
            .iter()
            .any(|p| matches!(p.kind, ParameterKind::Rest));

        let func_type = Type::Function {
            params: param_infos,
            return_type: Box::new(self.return_type.clone()),
            is_varargs,
            has_named_params,
            has_default_params,
            async_kind: crate::type_registry::AsyncKind::default(),
            calling_convention: crate::type_registry::CallingConvention::default(),
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        let body = TypedBlock {
            statements: self.statements,
            span: self.span,
        };

        let func = typed_node(
            TypedDeclaration::Function(TypedFunction {
                name: self.name,
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params: self.params,
                return_type: self.return_type,
                body: Some(body),
                visibility: self.visibility,
                is_async: self.is_async,
                is_pure: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
            }),
            func_type.clone(),
            self.span,
        );

        (func, func_type)
    }

    /// Consume the builder and return the inner TypedASTBuilder
    pub fn finish(self) -> TypedASTBuilder {
        self.builder
    }
}

/// Helper enum for default values that can be converted to expressions
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultValue {
    Integer(i128),
    Bool(bool),
    String(String),
    Unit,
}

impl DefaultValue {
    fn to_expression(
        self,
        builder: &mut TypedASTBuilder,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        match self {
            DefaultValue::Integer(i) => builder.int_literal(i, span),
            DefaultValue::Bool(b) => builder.bool_literal(b, span),
            DefaultValue::String(s) => builder.string_literal(&s, span),
            DefaultValue::Unit => builder.unit_literal(span),
        }
    }
}

impl From<i128> for DefaultValue {
    fn from(value: i128) -> Self {
        DefaultValue::Integer(value)
    }
}

impl From<i32> for DefaultValue {
    fn from(value: i32) -> Self {
        DefaultValue::Integer(value as i128)
    }
}

impl From<bool> for DefaultValue {
    fn from(value: bool) -> Self {
        DefaultValue::Bool(value)
    }
}

impl From<String> for DefaultValue {
    fn from(value: String) -> Self {
        DefaultValue::String(value)
    }
}

impl From<&str> for DefaultValue {
    fn from(value: &str) -> Self {
        DefaultValue::String(value.to_string())
    }
}

impl From<()> for DefaultValue {
    fn from(_: ()) -> Self {
        DefaultValue::Unit
    }
}

/// Extension methods for TypedASTBuilder to enable fluent API
impl TypedASTBuilder {
    /// Start building a function with fluent API
    pub fn function_builder(self, name: &str, span: Span) -> FluentFunctionBuilder {
        FluentFunctionBuilder::new(self, name, span)
    }

    /// Create a simple variable declaration and expression in one go
    pub fn var_with_ref(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        init_value: Option<TypedNode<TypedExpression>>,
        span: Span,
    ) -> (TypedNode<TypedStatement>, TypedNode<TypedExpression>) {
        let decl = self.let_statement(name, ty.clone(), mutability, init_value, span);
        let var_ref = self.variable(name, ty, span);
        (decl, var_ref)
    }
}

/// Helper trait to make PrimitiveType convertible to Type
impl From<PrimitiveType> for Type {
    fn from(prim: PrimitiveType) -> Self {
        Type::Primitive(prim)
    }
}

impl Default for TypedASTBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_expression_building() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 10);

        // Test various literal types
        let int_lit = builder.int_literal(42, span);
        let str_lit = builder.string_literal("hello", span);
        let bool_lit = builder.bool_literal(true, span);
        let char_lit = builder.char_literal('a', span);
        let unit_lit = builder.unit_literal(span);

        // Test expressions
        let var = builder.variable("x", Type::Primitive(PrimitiveType::I32), span);
        let binary = builder.binary(
            BinaryOp::Add,
            int_lit.clone(),
            var.clone(),
            Type::Primitive(PrimitiveType::I32),
            span,
        );

        // Verify structure
        assert!(matches!(
            int_lit.node,
            TypedExpression::Literal(TypedLiteral::Integer(42))
        ));
        assert!(matches!(binary.node, TypedExpression::Binary(_)));
    }

    #[test]
    fn test_function_call_with_named_arguments() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 10);

        let callee = builder.variable("func", Type::Primitive(PrimitiveType::I32), span);
        let arg1 = builder.int_literal(1, span);
        let arg2 = builder.int_literal(2, span);

        let call = builder.call_named(
            callee,
            vec![("x", arg1), ("y", arg2)],
            Type::Primitive(PrimitiveType::I32),
            span,
        );

        if let TypedExpression::Call(TypedCall { named_args, .. }) = &call.node {
            assert_eq!(named_args.len(), 2);
        } else {
            panic!("Expected function call with named arguments");
        }
    }

    #[test]
    fn test_enhanced_parameter_building() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 10);

        // Test different parameter types
        let regular = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let opt_default = builder.int_literal(42, span);
        let optional = builder.optional_parameter(
            "y",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            opt_default,
            span,
        );
        let rest = builder.rest_parameter(
            "args",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let out = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);

        assert_eq!(regular.kind, ParameterKind::Regular);
        assert_eq!(optional.kind, ParameterKind::Optional);
        assert_eq!(rest.kind, ParameterKind::Rest);
        assert_eq!(out.kind, ParameterKind::Out);
        assert!(optional.default_value.is_some());
    }

    #[test]
    fn test_pattern_matching_builders() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 10);

        // Test different pattern types
        let px_name = builder.intern("px");
        let py_name = builder.intern("py");
        let struct_pattern = builder.struct_pattern(
            "Point",
            vec![
                (
                    "x",
                    typed_node(TypedPattern::immutable_var(px_name), Type::Never, span),
                ),
                (
                    "y",
                    typed_node(TypedPattern::immutable_var(py_name), Type::Never, span),
                ),
            ],
            span,
        );

        let value_name = builder.intern("value");
        let enum_pattern = builder.enum_pattern(
            "Option",
            "Some",
            vec![typed_node(
                TypedPattern::immutable_var(value_name),
                Type::Never,
                span,
            )],
            span,
        );

        let first_name = builder.intern("first");
        let array_pattern = builder.array_pattern(
            vec![
                typed_node(TypedPattern::immutable_var(first_name), Type::Never, span),
                typed_node(TypedPattern::wildcard(), Type::Never, span),
            ],
            span,
        );

        // Verify patterns
        assert!(matches!(struct_pattern.node, TypedPattern::Struct { .. }));
        assert!(matches!(enum_pattern.node, TypedPattern::Enum { .. }));
        assert!(matches!(array_pattern.node, TypedPattern::Array(_)));
    }

    #[test]
    fn test_coroutine_building() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 10);

        let body = builder.int_literal(42, span);
        let coroutine = builder.coroutine(CoroutineKind::Async, body, vec![], span);

        if let TypedStatement::Coroutine(TypedCoroutine { kind, .. }) = &coroutine.node {
            assert_eq!(*kind, CoroutineKind::Async);
        } else {
            panic!("Expected coroutine statement");
        }
    }

    #[test]
    fn test_comprehensive_program_building() {
        let mut builder = TypedASTBuilder::new();
        let span = Span::new(0, 100);

        // Build a simple function with enhanced parameters
        let param1 = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let default_val = builder.int_literal(0, span);
        let param2 = builder.optional_parameter(
            "y",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            default_val,
            span,
        );

        let var_x = builder.variable("x", Type::Primitive(PrimitiveType::I32), span);
        let var_y = builder.variable("y", Type::Primitive(PrimitiveType::I32), span);
        let binary_expr = builder.binary(
            BinaryOp::Add,
            var_x,
            var_y,
            Type::Primitive(PrimitiveType::I32),
            span,
        );
        let body_stmt = builder.expression_statement(binary_expr, span);

        let body = builder.block(vec![body_stmt], span);

        let function = builder.function(
            "add",
            vec![param1, param2],
            Type::Primitive(PrimitiveType::I32),
            body,
            Visibility::Public,
            false,
            span,
        );

        let program = builder.program(vec![function], span);

        assert_eq!(program.declarations.len(), 1);
        if let TypedDeclaration::Function(func) = &program.declarations[0].node {
            assert_eq!(func.params.len(), 2);
            assert_eq!(func.params[1].kind, ParameterKind::Optional);
        }
    }

    #[test]
    fn test_new_statement_builders() {
        let mut builder = TypedASTBuilder::new();
        let span = builder.dummy_span();

        // Test return statement
        let value = builder.int_literal(42, span);
        let ret = builder.return_stmt(value, span);
        assert!(matches!(ret.node, TypedStatement::Return(Some(_))));

        // Test return void
        let ret_void = builder.return_void(span);
        assert!(matches!(ret_void.node, TypedStatement::Return(None)));

        // Test break
        let brk = builder.break_stmt(span);
        assert!(matches!(brk.node, TypedStatement::Break(None)));

        // Test break with value
        let break_val = builder.int_literal(100, span);
        let brk_with_val = builder.break_with_value(break_val, span);
        assert!(matches!(brk_with_val.node, TypedStatement::Break(Some(_))));

        // Test continue
        let cont = builder.continue_stmt(span);
        assert!(matches!(cont.node, TypedStatement::Continue));

        // Test throw
        let exception = builder.string_literal("error", span);
        let throw = builder.throw_stmt(exception, span);
        assert!(matches!(throw.node, TypedStatement::Throw(_)));

        // Test infinite loop
        let body = builder.block(vec![], span);
        let loop_stmt = builder.loop_stmt(body, span);
        assert!(matches!(
            loop_stmt.node,
            TypedStatement::Loop(TypedLoop::Infinite { .. })
        ));
    }

    #[test]
    fn test_new_expression_builders() {
        let mut builder = TypedASTBuilder::new();
        let span = builder.dummy_span();

        // Test cast
        let value = builder.int_literal(42, span);
        let cast = builder.cast(value, builder.i64_type(), span);
        assert!(matches!(cast.node, TypedExpression::Cast(_)));
        assert_eq!(cast.ty, builder.i64_type());

        // Test try
        let fallible = builder.variable("result", builder.i32_type(), span);
        let try_expr = builder.try_expr(fallible, builder.i32_type(), span);
        assert!(matches!(try_expr.node, TypedExpression::Try(_)));

        // Test await
        let future = builder.variable("async_val", builder.i32_type(), span);
        let await_expr = builder.await_expr(future, builder.i32_type(), span);
        assert!(matches!(await_expr.node, TypedExpression::Await(_)));

        // Test reference
        let var = builder.variable("x", builder.i32_type(), span);
        let ref_type = Type::Reference {
            ty: Box::new(builder.i32_type()),
            mutability: Mutability::Immutable,
            lifetime: None,
            nullability: crate::type_registry::NullabilityKind::NonNull,
        };
        let ref_expr = builder.reference(var, Mutability::Immutable, ref_type.clone(), span);
        assert!(matches!(ref_expr.node, TypedExpression::Reference(_)));
        assert_eq!(ref_expr.ty, ref_type);

        // Test dereference
        let ptr = builder.variable("ptr", ref_type.clone(), span);
        let deref = builder.dereference(ptr, builder.i32_type(), span);
        assert!(matches!(deref.node, TypedExpression::Dereference(_)));

        // Test range
        let start = builder.int_literal(0, span);
        let end = builder.int_literal(10, span);
        let range = builder.range(Some(start), Some(end), false, builder.i32_type(), span);
        assert!(matches!(range.node, TypedExpression::Range(_)));

        // Test tuple
        let elem1 = builder.int_literal(1, span);
        let elem2 = builder.string_literal("hello", span);
        let tuple_type = Type::Tuple(vec![builder.i32_type(), builder.string_type()]);
        let tuple = builder.tuple(vec![elem1, elem2], tuple_type.clone(), span);
        assert!(matches!(tuple.node, TypedExpression::Tuple(_)));
        assert_eq!(tuple.ty, tuple_type);

        // Test array literal
        let arr_elem1 = builder.int_literal(1, span);
        let arr_elem2 = builder.int_literal(2, span);
        let array_type = Type::Array {
            element_type: Box::new(builder.i32_type()),
            size: Some(crate::type_registry::ConstValue::Int(2)),
            nullability: crate::type_registry::NullabilityKind::NonNull,
        };
        let array = builder.array_literal(vec![arr_elem1, arr_elem2], array_type.clone(), span);
        assert!(matches!(array.node, TypedExpression::Array(_)));
        assert_eq!(array.ty, array_type);
    }

    #[test]
    fn test_type_helpers() {
        let builder = TypedASTBuilder::new();

        assert_eq!(builder.i32_type(), Type::Primitive(PrimitiveType::I32));
        assert_eq!(builder.i64_type(), Type::Primitive(PrimitiveType::I64));
        assert_eq!(builder.bool_type(), Type::Primitive(PrimitiveType::Bool));
        assert_eq!(
            builder.string_type(),
            Type::Primitive(PrimitiveType::String)
        );
        assert_eq!(builder.unit_type(), Type::Primitive(PrimitiveType::Unit));
        assert_eq!(builder.char_type(), Type::Primitive(PrimitiveType::Char));
        assert_eq!(builder.f32_type(), Type::Primitive(PrimitiveType::F32));
        assert_eq!(builder.f64_type(), Type::Primitive(PrimitiveType::F64));
    }

    #[test]
    fn test_span_helpers() {
        let builder = TypedASTBuilder::new();

        let span = builder.span(10, 20);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);

        let dummy = builder.dummy_span();
        assert_eq!(dummy.start, 0);
        assert_eq!(dummy.end, 0);
    }

    #[test]
    fn test_type_registry_integration() {
        let builder = TypedASTBuilder::new();

        // Verify that builder has a TypeRegistry
        // The registry is public and accessible
        let _ = &builder.registry;

        // Verify types can be created from the registry
        let i32_ty = builder.i32_type();
        assert_eq!(i32_ty, Type::Primitive(PrimitiveType::I32));
    }
}
