//! # Test DSL for TypedAST
//!
//! A fluent domain-specific language for building TypedAST structures in tests.
//! This provides a more elegant and less verbose way to construct test cases.

use crate::arena::{AstArena, InternedString};
use crate::source::Span;
use crate::type_registry::{Mutability, ParamInfo, PrimitiveType, Type, Visibility};
use crate::typed_builder::TypedASTBuilder;
use crate::{typed_ast::*, ConstValue};
use std::collections::HashMap;

/// A test context that tracks declarations and provides fluent building
pub struct TestContext {
    builder: TypedASTBuilder,
    /// Track function declarations for easy reference
    functions: HashMap<String, (TypedNode<TypedDeclaration>, Type)>,
    /// Track variable declarations for easy reference
    variables: HashMap<String, (TypedNode<TypedStatement>, Type)>,
    /// Default span for convenience
    default_span: Span,
}

impl TestContext {
    pub fn new(span: Span) -> Self {
        Self {
            builder: TypedASTBuilder::new(),
            functions: HashMap::new(),
            variables: HashMap::new(),
            default_span: span,
        }
    }

    /// Define a function and automatically track it
    pub fn define_function(&mut self, name: &str) -> FunctionDefiner {
        FunctionDefiner::new(self, name.to_string())
    }

    /// Reference a previously defined function
    pub fn function_ref(&mut self, name: &str) -> TypedNode<TypedExpression> {
        let (_, func_type) = self
            .functions
            .get(name)
            .expect(&format!("Function '{}' not defined", name));
        self.builder
            .variable(name, func_type.clone(), self.default_span)
    }

    /// Define a variable with optional initialization
    pub fn define_var(&mut self, name: &str, ty: impl Into<TypeDef>) -> VarDefiner {
        VarDefiner::new(self, name.to_string(), ty.into())
    }

    /// Reference a previously defined variable
    pub fn var_ref(&mut self, name: &str) -> TypedNode<TypedExpression> {
        let (_, var_type) = self
            .variables
            .get(name)
            .expect(&format!("Variable '{}' not defined", name));
        self.builder
            .variable(name, var_type.clone(), self.default_span)
    }

    /// Create a literal value
    pub fn literal(&mut self, value: impl Into<Literal>) -> TypedNode<TypedExpression> {
        value
            .into()
            .to_expression(&mut self.builder, self.default_span)
    }

    /// Create a function call
    pub fn call(&mut self, func_name: &str) -> CallBuilder {
        let func_ref = self.function_ref(func_name);
        CallBuilder::new(self, func_ref)
    }

    /// Build a program from all defined functions
    pub fn build_program(&mut self) -> TypedProgram {
        let mut declarations = Vec::new();

        // Add all functions
        for (_, (func_decl, _)) in &self.functions {
            declarations.push(func_decl.clone());
        }

        // Create main function if needed
        if !self.functions.contains_key("main") {
            let main_stmts: Vec<TypedNode<TypedStatement>> = self
                .variables
                .values()
                .map(|(stmt, _)| stmt.clone())
                .collect();

            let main_body = self.builder.block(main_stmts, self.default_span);
            let main_func = self.builder.function(
                "main",
                vec![],
                Type::Primitive(PrimitiveType::Unit),
                main_body,
                Visibility::Public,
                false,
                self.default_span,
            );
            declarations.push(main_func);
        }

        self.builder.program(declarations, self.default_span)
    }

    /// Get mutable access to the builder for advanced operations
    pub fn builder(&mut self) -> &mut TypedASTBuilder {
        &mut self.builder
    }
}

/// Fluent builder for function definitions
pub struct FunctionDefiner<'a> {
    ctx: &'a mut TestContext,
    name: String,
    params: Vec<ParamDef>,
    return_type: Type,
    statements: Vec<TypedNode<TypedStatement>>,
}

struct ParamDef {
    name: String,
    ty: Type,
    kind: ParamKind,
}

enum ParamKind {
    Regular(Mutability),
    Optional(TypedNode<TypedExpression>),
    Rest,
    Out,
    Ref(Mutability),
    InOut,
}

impl<'a> FunctionDefiner<'a> {
    fn new(ctx: &'a mut TestContext, name: String) -> Self {
        Self {
            ctx,
            name,
            params: Vec::new(),
            return_type: Type::Primitive(PrimitiveType::Unit),
            statements: Vec::new(),
        }
    }

    /// Add a parameter
    pub fn param(mut self, name: &str, ty: impl Into<TypeDef>) -> Self {
        self.params.push(ParamDef {
            name: name.to_string(),
            ty: ty.into().to_type(),
            kind: ParamKind::Regular(Mutability::Immutable),
        });
        self
    }

    /// Add an optional parameter
    pub fn optional(
        mut self,
        name: &str,
        ty: impl Into<TypeDef>,
        default: impl Into<Literal>,
    ) -> Self {
        let default_expr = default
            .into()
            .to_expression(&mut self.ctx.builder, self.ctx.default_span);
        self.params.push(ParamDef {
            name: name.to_string(),
            ty: ty.into().to_type(),
            kind: ParamKind::Optional(default_expr),
        });
        self
    }

    /// Add a rest parameter
    pub fn rest(mut self, name: &str, ty: impl Into<TypeDef>) -> Self {
        self.params.push(ParamDef {
            name: name.to_string(),
            ty: ty.into().to_type(),
            kind: ParamKind::Rest,
        });
        self
    }

    /// Add an out parameter
    pub fn out(mut self, name: &str, ty: impl Into<TypeDef>) -> Self {
        self.params.push(ParamDef {
            name: name.to_string(),
            ty: ty.into().to_type(),
            kind: ParamKind::Out,
        });
        self
    }

    /// Set return type
    pub fn returns(mut self, ty: impl Into<TypeDef>) -> Self {
        self.return_type = ty.into().to_type();
        self
    }

    /// Add a statement to the function body
    pub fn stmt(mut self, stmt: TypedNode<TypedStatement>) -> Self {
        self.statements.push(stmt);
        self
    }

    /// Complete the function definition
    pub fn end(self) -> &'a mut TestContext {
        let span = self.ctx.default_span;
        let builder = &mut self.ctx.builder;

        // Convert params to TypedParameter
        let typed_params: Vec<TypedParameter> = self
            .params
            .into_iter()
            .map(|p| match p.kind {
                ParamKind::Regular(mutability) => {
                    builder.parameter(&p.name, p.ty, mutability, span)
                }
                ParamKind::Optional(default) => {
                    builder.optional_parameter(&p.name, p.ty, Mutability::Immutable, default, span)
                }
                ParamKind::Rest => {
                    builder.rest_parameter(&p.name, p.ty, Mutability::Immutable, span)
                }
                ParamKind::Out => builder.out_parameter(&p.name, p.ty, span),
                ParamKind::Ref(mutability) => {
                    builder.ref_parameter(&p.name, p.ty, mutability, span)
                }
                ParamKind::InOut => builder.inout_parameter(&p.name, p.ty, span),
            })
            .collect();

        // Build function type
        let param_infos: Vec<ParamInfo> = typed_params
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

        let func_type = Type::Function {
            params: param_infos,
            return_type: Box::new(self.return_type.clone()),
            is_varargs: typed_params
                .iter()
                .any(|p| matches!(p.kind, ParameterKind::Rest)),
            has_named_params: true,
            has_default_params: typed_params
                .iter()
                .any(|p| matches!(p.kind, ParameterKind::Optional)),
            async_kind: crate::AsyncKind::Sync,
            calling_convention: crate::CallingConvention::Default,
            nullability: crate::NullabilityKind::Unknown,
        };

        // Build function
        let body = builder.block(self.statements, span);
        let func_decl = builder.function(
            &self.name,
            typed_params,
            self.return_type,
            body,
            Visibility::Public,
            false,
            span,
        );

        // Store in context
        self.ctx.functions.insert(self.name, (func_decl, func_type));
        self.ctx
    }
}

/// Fluent builder for variable definitions
pub struct VarDefiner<'a> {
    ctx: &'a mut TestContext,
    name: String,
    ty: Type,
    mutability: Mutability,
    init: Option<TypedNode<TypedExpression>>,
}

impl<'a> VarDefiner<'a> {
    fn new(ctx: &'a mut TestContext, name: String, ty_def: TypeDef) -> Self {
        Self {
            ctx,
            name,
            ty: ty_def.to_type(),
            mutability: Mutability::Immutable,
            init: None,
        }
    }

    /// Make the variable mutable
    pub fn mutable(mut self) -> Self {
        self.mutability = Mutability::Mutable;
        self
    }

    /// Initialize with a value
    pub fn init(mut self, value: impl Into<Literal>) -> Self {
        self.init = Some(
            value
                .into()
                .to_expression(&mut self.ctx.builder, self.ctx.default_span),
        );
        self
    }

    /// Initialize with an expression
    pub fn init_expr(mut self, expr: TypedNode<TypedExpression>) -> Self {
        self.init = Some(expr);
        self
    }

    /// Complete the variable definition
    pub fn end(self) -> &'a mut TestContext {
        let stmt = self.ctx.builder.let_statement(
            &self.name,
            self.ty.clone(),
            self.mutability,
            self.init,
            self.ctx.default_span,
        );

        self.ctx.variables.insert(self.name, (stmt, self.ty));
        self.ctx
    }
}

/// Fluent builder for function calls
pub struct CallBuilder<'a> {
    ctx: &'a mut TestContext,
    callee: TypedNode<TypedExpression>,
    positional_args: Vec<TypedNode<TypedExpression>>,
    named_args: Vec<(String, TypedNode<TypedExpression>)>,
    return_type: Type,
}

impl<'a> CallBuilder<'a> {
    fn new(ctx: &'a mut TestContext, callee: TypedNode<TypedExpression>) -> Self {
        let return_type = if let Type::Function { return_type, .. } = &callee.ty {
            (**return_type).clone()
        } else {
            Type::Never
        };

        Self {
            ctx,
            callee,
            positional_args: Vec::new(),
            named_args: Vec::new(),
            return_type,
        }
    }

    /// Add a positional argument
    pub fn arg(mut self, value: impl Into<Literal>) -> Self {
        let expr = value
            .into()
            .to_expression(&mut self.ctx.builder, self.ctx.default_span);
        self.positional_args.push(expr);
        self
    }

    /// Add a positional argument expression
    pub fn arg_expr(mut self, expr: TypedNode<TypedExpression>) -> Self {
        self.positional_args.push(expr);
        self
    }

    /// Add a named argument
    pub fn named(mut self, name: &str, value: impl Into<Literal>) -> Self {
        let expr = value
            .into()
            .to_expression(&mut self.ctx.builder, self.ctx.default_span);
        self.named_args.push((name.to_string(), expr));
        self
    }

    /// Build the call expression
    pub fn build(self) -> TypedNode<TypedExpression> {
        let span = self.ctx.default_span;

        if self.named_args.is_empty() {
            self.ctx.builder.call_positional(
                self.callee,
                self.positional_args,
                self.return_type,
                span,
            )
        } else if self.positional_args.is_empty() {
            let named_args: Vec<(&str, TypedNode<TypedExpression>)> = self
                .named_args
                .iter()
                .map(|(name, expr)| (name.as_str(), expr.clone()))
                .collect();
            self.ctx
                .builder
                .call_named(self.callee, named_args, self.return_type, span)
        } else {
            let named_args: Vec<(&str, TypedNode<TypedExpression>)> = self
                .named_args
                .iter()
                .map(|(name, expr)| (name.as_str(), expr.clone()))
                .collect();
            self.ctx.builder.call_mixed(
                self.callee,
                self.positional_args,
                named_args,
                vec![],
                self.return_type,
                span,
            )
        }
    }
}

/// Type definition that can be primitive or complex
#[derive(Debug, Clone)]
pub enum TypeDef {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Char,
    String,
    Unit,
    Array(Box<TypeDef>, Option<usize>),
    Named(String),
    Custom(Type),
}

impl TypeDef {
    fn to_type(self) -> Type {
        match self {
            TypeDef::I32 => Type::Primitive(PrimitiveType::I32),
            TypeDef::I64 => Type::Primitive(PrimitiveType::I64),
            TypeDef::F32 => Type::Primitive(PrimitiveType::F32),
            TypeDef::F64 => Type::Primitive(PrimitiveType::F64),
            TypeDef::Bool => Type::Primitive(PrimitiveType::Bool),
            TypeDef::Char => Type::Primitive(PrimitiveType::Char),
            TypeDef::String => Type::Primitive(PrimitiveType::String),
            TypeDef::Unit => Type::Primitive(PrimitiveType::Unit),
            TypeDef::Array(elem, size) => Type::Array {
                element_type: Box::new(elem.to_type()),
                size: size.map(|s| ConstValue::UInt(s as u64)),
                nullability: crate::NullabilityKind::Unknown,
            },
            TypeDef::Named(_name) => Type::Never, // Would need proper interning through builder
            TypeDef::Custom(ty) => ty,
        }
    }
}

// Convenient From implementations
impl From<PrimitiveType> for TypeDef {
    fn from(prim: PrimitiveType) -> Self {
        match prim {
            PrimitiveType::I32 => TypeDef::I32,
            PrimitiveType::I64 => TypeDef::I64,
            PrimitiveType::F32 => TypeDef::F32,
            PrimitiveType::F64 => TypeDef::F64,
            PrimitiveType::Bool => TypeDef::Bool,
            PrimitiveType::Char => TypeDef::Char,
            PrimitiveType::String => TypeDef::String,
            PrimitiveType::Unit => TypeDef::Unit,
            _ => TypeDef::Custom(Type::Primitive(prim)),
        }
    }
}

impl From<Type> for TypeDef {
    fn from(ty: Type) -> Self {
        TypeDef::Custom(ty)
    }
}

/// Literal values that can be easily converted
#[derive(Debug, Clone)]
pub enum Literal {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Char(char),
    Unit,
}

impl Literal {
    fn to_expression(
        self,
        builder: &mut TypedASTBuilder,
        span: Span,
    ) -> TypedNode<TypedExpression> {
        match self {
            Literal::Int(i) => builder.int_literal(i, span),
            Literal::Float(f) => typed_node(
                TypedExpression::Literal(TypedLiteral::Float(f)),
                Type::Primitive(PrimitiveType::F64),
                span,
            ),
            Literal::Bool(b) => builder.bool_literal(b, span),
            Literal::String(s) => builder.string_literal(&s, span),
            Literal::Char(c) => builder.char_literal(c, span),
            Literal::Unit => builder.unit_literal(span),
        }
    }
}

// Convenient From implementations for literals
impl From<i32> for Literal {
    fn from(value: i32) -> Self {
        Literal::Int(value as i128)
    }
}

impl From<i64> for Literal {
    fn from(value: i64) -> Self {
        Literal::Int(value as i128)
    }
}

impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Literal::Float(value as f64)
    }
}

impl From<f64> for Literal {
    fn from(value: f64) -> Self {
        Literal::Float(value)
    }
}

impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Literal::Bool(value)
    }
}

impl From<&str> for Literal {
    fn from(value: &str) -> Self {
        Literal::String(value.to_string())
    }
}

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Literal::String(value)
    }
}

impl From<char> for Literal {
    fn from(value: char) -> Self {
        Literal::Char(value)
    }
}

impl From<()> for Literal {
    fn from(_: ()) -> Self {
        Literal::Unit
    }
}
