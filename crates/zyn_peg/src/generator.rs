//! Code generator for ZynPEG
//!
//! Generates Rust code from .zyn grammars:
//! 1. A pest parser (from the PEG patterns)
//! 2. A TypedAST builder (from the action blocks)

use crate::error::{Result, ZynPegError};
use crate::{ActionBlock, RuleDef, RuleModifier, ZynGrammar};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::io::Write;
use std::process::{Command, Stdio};

/// Generate complete Rust code from a ZynGrammar
pub fn generate_parser(grammar: &ZynGrammar) -> Result<GeneratedCode> {
    let pest_grammar = generate_pest_grammar(grammar)?;
    let ast_builder = generate_ast_builder(grammar)?;
    let parser_impl = generate_parser_impl(grammar)?;

    Ok(GeneratedCode {
        pest_grammar,
        ast_builder,
        parser_impl,
        typed_ast_types: None,
    })
}

/// Generated code components
pub struct GeneratedCode {
    /// The .pest grammar file content
    pub pest_grammar: String,
    /// The TypedAST builder Rust code (TokenStream)
    pub ast_builder: TokenStream,
    /// The parser implementation with parse_to_typed_ast method (TokenStream)
    pub parser_impl: TokenStream,
    /// Optional standalone TypedAST types module (TokenStream)
    pub typed_ast_types: Option<TokenStream>,
}

impl GeneratedCode {
    /// Get formatted AST builder code using rustfmt
    pub fn ast_builder_formatted(&self) -> String {
        format_rust_code(&self.ast_builder.to_string())
    }

    /// Get formatted parser impl code using rustfmt
    pub fn parser_impl_formatted(&self) -> String {
        format_rust_code(&self.parser_impl.to_string())
    }

    /// Get formatted TypedAST types code using rustfmt
    pub fn typed_ast_types_formatted(&self) -> Option<String> {
        self.typed_ast_types
            .as_ref()
            .map(|ts| format_rust_code(&ts.to_string()))
    }
}

/// Generate complete standalone parser with TypedAST types
///
/// This generates a complete, self-contained parser that doesn't depend on
/// external type crates. It includes:
/// - typed_ast.rs: Complete TypedAST type definitions
/// - ast_builder.rs: Parser actions that build TypedAST
/// - parser_impl.rs: Convenience parse functions
pub fn generate_standalone_parser(grammar: &ZynGrammar) -> Result<GeneratedCode> {
    let pest_grammar = generate_pest_grammar(grammar)?;
    let ast_builder = generate_standalone_ast_builder(grammar)?;
    let parser_impl = generate_standalone_parser_impl(grammar)?;
    let typed_ast_types = Some(generate_typed_ast_types(grammar)?);

    Ok(GeneratedCode {
        pest_grammar,
        ast_builder,
        parser_impl,
        typed_ast_types,
    })
}

/// Generate parser that uses zyntax_typed_ast types directly
///
/// This generates a parser that integrates with the existing zyntax type system:
/// - Uses zyntax_typed_ast::Type, TypedExpression, TypedNode, etc.
/// - Uses InternedString for string interning
/// - Uses Type::Primitive(PrimitiveType::I32) format for types
/// - Compatible with zyntax_compiler for JIT execution
///
/// No typed_ast.rs is generated - the parser imports from zyntax_typed_ast.
pub fn generate_zyntax_parser(grammar: &ZynGrammar) -> Result<GeneratedCode> {
    let pest_grammar = generate_pest_grammar(grammar)?;
    let ast_builder = generate_zyntax_ast_builder(grammar)?;
    let parser_impl = generate_zyntax_parser_impl(grammar)?;

    Ok(GeneratedCode {
        pest_grammar,
        ast_builder,
        parser_impl,
        typed_ast_types: None, // Uses zyntax_typed_ast, no standalone types
    })
}

/// Generate the TypedAST types module for the grammar
///
/// This generates a complete, standalone TypedAST API including:
/// - Core types (Span, Type, Expression, Statement, Declaration)
/// - TypedProgram and related structures
/// - Helper functions for AST construction
fn generate_typed_ast_types(grammar: &ZynGrammar) -> Result<TokenStream> {
    let lang_name = &grammar.language.name;
    let lang_comment = format!("Generated for: {}", lang_name);

    Ok(quote! {
        //! ZynPEG TypedAST Types
        //!
        //! This module provides the complete TypedAST type system for the
        //! generated parser. These types form the semantic representation
        //! of parsed source code.
        //!
        #![doc = #lang_comment]

        #![allow(dead_code, unused_variables, unused_imports)]

        use std::collections::HashMap;

        // ============================================================
        // Core Types
        // ============================================================

        /// Span in source code
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct Span {
            pub start: usize,
            pub end: usize,
        }

        impl Span {
            pub fn new(start: usize, end: usize) -> Self {
                Self { start, end }
            }

            pub fn merge(a: &Self, b: &Self) -> Self {
                Self {
                    start: a.start.min(b.start),
                    end: a.end.max(b.end),
                }
            }
        }

        /// Type representation (simplified for testing)
        #[derive(Debug, Clone, PartialEq)]
        pub enum Type {
            // Primitives
            I8, I16, I32, I64, I128,
            U8, U16, U32, U64, U128,
            F32, F64,
            Bool,
            Void,
            String,
            Char,
            Usize, Isize,

            // Compound types
            Array(Box<Type>, Option<usize>),
            Slice(Box<Type>),
            Pointer(Box<Type>),
            Optional(Box<Type>),
            Result(Box<Type>, Box<Type>),
            Function(Vec<Type>, Box<Type>),
            Tuple(Vec<Type>),
            Named(String),
            Unknown,
            Never,
            Any,
            Comptime(Box<Type>),
        }

        impl Default for Type {
            fn default() -> Self { Type::Unknown }
        }

        impl Type {
            pub fn unwrap_optional(&self) -> Type {
                match self {
                    Type::Optional(inner) => (**inner).clone(),
                    _ => self.clone(),
                }
            }

            pub fn unwrap_result(&self) -> Type {
                match self {
                    Type::Result(inner, _) => (**inner).clone(),
                    _ => self.clone(),
                }
            }

            pub fn deref(&self) -> Type {
                match self {
                    Type::Pointer(inner) => (**inner).clone(),
                    _ => self.clone(),
                }
            }

            pub fn element_type(&self) -> Type {
                match self {
                    Type::Array(inner, _) | Type::Slice(inner) => (**inner).clone(),
                    _ => Type::Unknown,
                }
            }

            pub fn return_type(&self) -> Type {
                match self {
                    Type::Function(_, ret) => (**ret).clone(),
                    _ => Type::Unknown,
                }
            }
        }

        // ============================================================
        // Expression Types
        // ============================================================

        /// Binary operations
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum BinaryOp {
            // Arithmetic
            Add, Sub, Mul, Div, Mod,
            // Comparison
            Eq, Ne, Lt, Le, Gt, Ge,
            // Logical
            And, Or,
            // Bitwise
            BitAnd, BitOr, BitXor, Shl, Shr,
            // Special (Zig-specific)
            Orelse, Catch,
        }

        /// Unary operations
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum UnaryOp {
            Neg, Not, Deref, AddrOf, Try, Await,
        }

        /// Postfix operations for expression chaining
        #[derive(Debug, Clone, PartialEq)]
        pub enum PostfixOp {
            Deref,
            Field(String),
            Index(TypedExpression),
            Call(Vec<TypedExpression>),
            OptionalUnwrap,
            TryUnwrap,
        }

        /// A typed expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct TypedExpression {
            pub expr: Expression,
            pub ty: Type,
            pub span: Span,
        }

        impl Default for TypedExpression {
            fn default() -> Self {
                Self {
                    expr: Expression::Unit,
                    ty: Type::Unknown,
                    span: Span::default(),
                }
            }
        }

        /// Expression variants
        #[derive(Debug, Clone, PartialEq)]
        pub enum Expression {
            // Literals
            IntLiteral(i64),
            FloatLiteral(f64),
            StringLiteral(String),
            CharLiteral(char),
            BoolLiteral(bool),
            NullLiteral,
            UndefinedLiteral,

            // Variables and references
            Identifier(String),
            Variable(String, Type),

            // Operations
            BinaryOp(BinaryOp, Box<TypedExpression>, Box<TypedExpression>),
            UnaryOp(UnaryOp, Box<TypedExpression>),

            // Access
            FieldAccess(Box<TypedExpression>, String),
            Index(Box<TypedExpression>, Box<TypedExpression>),
            Deref(Box<TypedExpression>),

            // Calls
            Call(Box<TypedExpression>, Vec<TypedExpression>),
            MethodCall(Box<TypedExpression>, String, Vec<TypedExpression>),

            // Collections
            Array(Vec<TypedExpression>),
            Tuple(Vec<TypedExpression>),
            Struct(String, Vec<(String, TypedExpression)>),

            // Control flow as expression
            If(Box<TypedExpression>, Box<TypedExpression>, Option<Box<TypedExpression>>),
            Block(Vec<TypedStatement>),

            // Lambda/closures
            Lambda(Lambda),

            // Try expression
            Try(Try),

            // Switch expression
            Switch(Switch),

            // Struct literal expression
            StructLiteral(StructLiteral),

            // Special
            Unit,
            Error(String),
        }

        // ============================================================
        // Statement Types
        // ============================================================

        /// A typed statement
        #[derive(Debug, Clone, PartialEq)]
        pub struct TypedStatement {
            pub stmt: Statement,
            pub span: Span,
        }

        /// Statement variants
        #[derive(Debug, Clone, PartialEq)]
        pub enum Statement {
            Let(String, Option<Type>, Option<TypedExpression>),
            Const(String, Option<Type>, TypedExpression),
            Assign(TypedExpression, TypedExpression),
            Expression(TypedExpression),
            Return(Option<TypedExpression>),
            If(TypedExpression, Vec<TypedStatement>, Option<Vec<TypedStatement>>),
            While(TypedExpression, Vec<TypedStatement>),
            For(String, TypedExpression, Vec<TypedStatement>),
            Block(Vec<TypedStatement>),
            Break(Option<TypedExpression>),
            Continue,
            Defer(Box<TypedStatement>),
            Try(Box<TypedStatement>, Vec<CatchClause>),
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct CatchClause {
            pub error_var: String,
            pub body: Vec<TypedStatement>,
        }

        // ============================================================
        // Declaration Types
        // ============================================================

        /// A typed declaration
        #[derive(Debug, Clone, PartialEq)]
        pub struct TypedDeclaration {
            pub decl: Declaration,
            pub span: Span,
        }

        /// Declaration variants
        #[derive(Debug, Clone, PartialEq)]
        pub enum Declaration {
            Function(FnDecl),
            Const(ConstDecl),
            Var(VarDecl),
            Struct(StructDecl),
            Enum(EnumDecl),
            Union(UnionDecl),
            ErrorSet(ErrorSetDecl),
            Import(ImportDecl),
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct FnDecl {
            pub name: String,
            pub params: Vec<Param>,
            pub return_type: Type,
            pub body: Option<Vec<TypedStatement>>,
            pub is_pub: bool,
            pub is_export: bool,
            pub is_extern: bool,
        }

        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct Param {
            pub name: String,
            pub ty: Type,
            pub is_comptime: bool,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct ConstDecl {
            pub name: String,
            pub ty: Option<Type>,
            pub value: TypedExpression,
            pub is_pub: bool,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct VarDecl {
            pub name: String,
            pub ty: Option<Type>,
            pub value: Option<TypedExpression>,
            pub is_pub: bool,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct StructDecl {
            pub name: String,
            pub fields: Vec<FieldDecl>,
            pub is_packed: bool,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct FieldDecl {
            pub name: String,
            pub ty: Type,
            pub default: Option<TypedExpression>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct EnumDecl {
            pub name: String,
            pub tag_type: Option<Type>,
            pub variants: Vec<EnumVariant>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct EnumVariant {
            pub name: String,
            pub value: Option<TypedExpression>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct UnionDecl {
            pub name: String,
            pub is_tagged: bool,
            pub fields: Vec<UnionField>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct UnionField {
            pub name: String,
            pub ty: Type,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct ErrorSetDecl {
            pub name: String,
            pub errors: Vec<String>,
        }

        #[derive(Debug, Clone, PartialEq)]
        pub struct ImportDecl {
            pub path: String,
        }

        // ============================================================
        // Program
        // ============================================================

        /// A complete program
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct TypedProgram {
            pub declarations: Vec<TypedDeclaration>,
            pub span: Span,
        }

        // ============================================================
        // Helper Functions
        // ============================================================

        /// Create span from two HasSpan items
        pub fn make_span(start_pair: impl HasSpan, end_pair: impl HasSpan) -> Span {
            Span::merge(&start_pair.span(), &end_pair.span())
        }

        pub trait HasSpan {
            fn span(&self) -> Span;
        }

        impl HasSpan for Span {
            fn span(&self) -> Span { *self }
        }

        impl HasSpan for TypedExpression {
            fn span(&self) -> Span { self.span }
        }

        impl HasSpan for TypedStatement {
            fn span(&self) -> Span { self.span }
        }

        impl HasSpan for TypedDeclaration {
            fn span(&self) -> Span { self.span }
        }

        /// Intern a string (stub - just returns the string)
        pub fn intern(s: &str) -> String {
            s.to_string()
        }

        /// Parse integer from string
        pub fn parse_int(s: &str) -> i64 {
            s.parse().unwrap_or(0)
        }

        /// Parse float from string
        pub fn parse_float(s: &str) -> f64 {
            s.parse().unwrap_or(0.0)
        }

        /// Unwrap a result type to get the inner type
        pub fn unwrap_result_type(ty: Type) -> Type {
            match ty {
                Type::Result(inner, _) => *inner,
                _ => ty,
            }
        }

        /// Convert a TypedExpression to TypedLiteral (for pattern matching)
        pub fn expr_to_literal(expr: TypedExpression) -> TypedLiteral {
            match expr.expr {
                Expression::IntLiteral(n) => TypedLiteral::Int(n),
                Expression::FloatLiteral(f) => TypedLiteral::Float(f),
                Expression::StringLiteral(s) => TypedLiteral::String(s),
                Expression::CharLiteral(c) => TypedLiteral::Char(c),
                Expression::BoolLiteral(b) => TypedLiteral::Bool(b),
                Expression::NullLiteral => TypedLiteral::Null,
                Expression::UndefinedLiteral => TypedLiteral::Undefined,
                _ => TypedLiteral::Null, // Fallback for non-literal expressions
            }
        }

        // ============================================================
        // Error Types
        // ============================================================

        /// Parse error
        #[derive(Debug)]
        pub struct ParseError(pub String);

        impl std::fmt::Display for ParseError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl std::error::Error for ParseError {}

        // Specific impl for pest errors to avoid blanket impl conflict
        impl<R: pest::RuleType> From<pest::error::Error<R>> for ParseError {
            fn from(e: pest::error::Error<R>) -> Self {
                ParseError(e.to_string())
            }
        }

        // ============================================================
        // Type Aliases for Compatibility with Action Blocks
        // ============================================================

        /// Alias for Param - used in action blocks as TypedParam
        pub type TypedParam = Param;

        /// Alias for FieldDecl - used in action blocks as TypedField
        pub type TypedField = FieldDecl;

        /// Alias for FnDecl - used in action blocks as FunctionDecl
        pub type FunctionDecl = FnDecl;

        /// TypedBlock - wrapper for a block of statements
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct TypedBlock {
            pub statements: Vec<TypedStatement>,
            pub span: Span,
        }

        /// TypedLiteral for literal values
        #[derive(Debug, Clone, PartialEq)]
        pub enum TypedLiteral {
            Int(i64),
            Float(f64),
            String(String),
            Char(char),
            Bool(bool),
            Null,
            Undefined,
        }

        // ============================================================
        // Visibility
        // ============================================================

        /// Visibility modifier for declarations
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub enum Visibility {
            #[default]
            Private,
            Public,
        }

        // ============================================================
        // Patterns for Pattern Matching
        // ============================================================

        /// Pattern for pattern matching
        #[derive(Debug, Clone, PartialEq)]
        pub enum Pattern {
            Wildcard,
            Binding(String),
            Literal(TypedLiteral),
            Variant(String, Box<Pattern>),
            Tuple(Vec<Pattern>),
            Struct(String, Vec<(String, Pattern)>),
        }

        impl Default for Pattern {
            fn default() -> Self { Pattern::Wildcard }
        }

        /// Match arm for switch/match expressions
        #[derive(Debug, Clone, PartialEq)]
        pub struct MatchArm {
            pub pattern: Pattern,
            pub body: TypedExpression,
        }

        // ============================================================
        // Span from pest
        // ============================================================

        impl Span {
            /// Create a Span from a pest::Span
            pub fn from_pest<'i>(pest_span: pest::Span<'i>) -> Self {
                Self {
                    start: pest_span.start(),
                    end: pest_span.end(),
                }
            }
        }

        // ============================================================
        // Helper Functions for Action Blocks
        // ============================================================

        /// Infer type from an expression (stub)
        pub fn infer_type<T>(_expr: T) -> Type {
            Type::Unknown
        }

        /// Parse assignment operator from string
        pub fn parse_assign_op<T>(_op: T) -> AssignOp {
            AssignOp::Assign
        }

        /// Parse equality operator
        pub fn parse_eq_op<T>(_op: T) -> BinaryOp {
            BinaryOp::Eq
        }

        /// Parse comparison operator
        pub fn parse_cmp_op<T>(_op: T) -> BinaryOp {
            BinaryOp::Lt
        }

        /// Parse shift operator
        pub fn parse_shift_op<T>(_op: T) -> BinaryOp {
            BinaryOp::Shl
        }

        /// Parse add operator
        pub fn parse_add_op<T>(_op: T) -> BinaryOp {
            BinaryOp::Add
        }

        /// Parse mul operator
        pub fn parse_mul_op<T>(_op: T) -> BinaryOp {
            BinaryOp::Mul
        }

        /// Parse unary operator
        pub fn parse_unary_op<T>(_op: T) -> UnaryOp {
            UnaryOp::Neg
        }

        /// Fold binary operations
        pub fn fold_binary(first: TypedExpression, rest: Vec<TypedExpression>, default_op: BinaryOp) -> TypedExpression {
            if rest.is_empty() {
                return first;
            }
            rest.into_iter().fold(first, |left, right| {
                let result_ty = match default_op {
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le |
                    BinaryOp::Gt | BinaryOp::Ge | BinaryOp::And | BinaryOp::Or => Type::Bool,
                    BinaryOp::Orelse => left.ty.unwrap_optional(),
                    BinaryOp::Catch => left.ty.unwrap_result(),
                    _ => left.ty.clone(),
                };
                TypedExpression {
                    expr: Expression::BinaryOp(default_op, Box::new(left.clone()), Box::new(right.clone())),
                    ty: result_ty,
                    span: Span::merge(&left.span, &right.span),
                }
            })
        }

        /// Fold postfix operations
        pub fn fold_postfix(base: TypedExpression, ops: Vec<PostfixOp>) -> TypedExpression {
            ops.into_iter().fold(base, |expr, op| {
                let span = expr.span.clone();
                match op {
                    PostfixOp::Deref => TypedExpression {
                        ty: expr.ty.deref(),
                        expr: Expression::Deref(Box::new(expr)),
                        span,
                    },
                    PostfixOp::Field(name) => TypedExpression {
                        ty: Type::Unknown,
                        expr: Expression::FieldAccess(Box::new(expr), name),
                        span,
                    },
                    PostfixOp::Index(index) => TypedExpression {
                        ty: expr.ty.element_type(),
                        expr: Expression::Index(Box::new(expr), Box::new(index)),
                        span,
                    },
                    PostfixOp::Call(args) => TypedExpression {
                        ty: expr.ty.return_type(),
                        expr: Expression::Call(Box::new(expr), args),
                        span,
                    },
                    PostfixOp::OptionalUnwrap => TypedExpression {
                        ty: expr.ty.unwrap_optional(),
                        expr: Expression::UnaryOp(UnaryOp::Try, Box::new(expr)),
                        span,
                    },
                    PostfixOp::TryUnwrap => TypedExpression {
                        ty: expr.ty.unwrap_result(),
                        expr: Expression::UnaryOp(UnaryOp::Try, Box::new(expr)),
                        span,
                    },
                }
            })
        }

        /// Collect switch cases
        pub fn collect_cases<T, U>(_scrutinee: T, _cases: U) -> Vec<(Pattern, TypedExpression)> {
            vec![]
        }

        /// Infer switch result type
        pub fn infer_switch_type<T, U>(_scrutinee: T, _cases: U) -> Type {
            Type::Unknown
        }

        /// Collect function parameters
        pub fn collect_params(first: Param, rest: Vec<Param>) -> Vec<Param> {
            let mut result = vec![first];
            result.extend(rest);
            result
        }

        /// Collect struct fields
        pub fn collect_fields<T, U>(_first: T, _rest: U) -> Vec<(String, TypedExpression)> {
            vec![]
        }

        /// Collect expressions into a vector
        pub fn collect_exprs(first: TypedExpression, rest: Vec<TypedExpression>) -> Vec<TypedExpression> {
            let mut result = vec![first];
            result.extend(rest);
            result
        }

        /// Assignment operators
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub enum AssignOp {
            #[default]
            Assign,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
            ModAssign,
            BitAndAssign,
            BitOrAssign,
            BitXorAssign,
            ShlAssign,
            ShrAssign,
        }

        /// Error type
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct Error;

        impl Type {
            /// The Error type constant
            #[allow(non_upper_case_globals)]
            pub const Error: Type = Type::Unknown;
            /// The Type type constant (for comptime)
            #[allow(non_upper_case_globals)]
            pub const Type: Type = Type::Unknown;
        }

        // ============================================================
        // Statement Struct Types (for direct use in action blocks)
        // ============================================================

        /// Assignment statement (can be used directly in stmt field)
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct Assignment {
            pub target: TypedExpression,
            pub op: AssignOp,
            pub value: TypedExpression,
        }

        impl From<Assignment> for Statement {
            fn from(a: Assignment) -> Self {
                Statement::Assign(a.target, a.value)
            }
        }

        /// Return statement
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct Return {
            pub value: Option<TypedExpression>,
        }

        impl From<Return> for Statement {
            fn from(r: Return) -> Self {
                Statement::Return(r.value)
            }
        }

        /// Expression statement
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct ExprStmt {
            pub expr: TypedExpression,
        }

        impl From<ExprStmt> for Statement {
            fn from(e: ExprStmt) -> Self {
                Statement::Expression(e.expr)
            }
        }

        /// Match statement (pattern matching)
        #[derive(Debug, Clone, PartialEq)]
        pub struct Match {
            pub scrutinee: TypedExpression,
            pub arms: Vec<MatchArm>,
        }

        impl Default for Match {
            fn default() -> Self {
                Self { scrutinee: TypedExpression::default(), arms: vec![] }
            }
        }

        /// If statement
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct If {
            pub condition: TypedExpression,
            pub then_branch: TypedBlock,
            pub else_branch: Option<TypedBlock>,
        }

        /// While statement
        #[derive(Debug, Clone, PartialEq, Default)]
        pub struct While {
            pub condition: TypedExpression,
            pub body: TypedBlock,
        }

        /// Break statement (unit struct for direct use)
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct Break;

        /// Continue statement (unit struct for direct use)
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct Continue;

        // ============================================================
        // Expression Struct Types (for direct use in action blocks)
        // ============================================================

        /// Literal expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct Literal(pub TypedLiteral);

        /// UnaryOp expression (for action blocks)
        #[derive(Debug, Clone, PartialEq)]
        pub struct UnaryOpExpr(pub UnaryOp, pub Box<TypedExpression>);

        /// Array expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct Array(pub Vec<TypedExpression>);

        /// Struct literal expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct StructLiteral {
            pub name: String,
            pub fields: Vec<(String, TypedExpression)>,
        }

        /// Lambda expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct Lambda {
            pub params: Vec<TypedParam>,
            pub body: TypedBlock,
            pub captures: Vec<String>,
        }

        impl Default for Lambda {
            fn default() -> Self {
                Self { params: vec![], body: TypedBlock::default(), captures: vec![] }
            }
        }

        /// Try expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct Try {
            pub expr: Box<TypedExpression>,
        }

        impl Default for Try {
            fn default() -> Self {
                Self { expr: Box::new(TypedExpression::default()) }
            }
        }

        /// Switch expression
        #[derive(Debug, Clone, PartialEq)]
        pub struct Switch {
            pub scrutinee: Box<TypedExpression>,
            pub cases: Vec<(Pattern, TypedExpression)>,
        }

        impl Default for Switch {
            fn default() -> Self {
                Self { scrutinee: Box::new(TypedExpression::default()), cases: vec![] }
            }
        }

        // ============================================================
        // Helper Constructors for Expression Variants
        // These allow action blocks to use short names like `UnaryOp(...)`
        // instead of `Expression::UnaryOp(...)`
        // ============================================================

        /// Construct Expression::UnaryOp variant
        /// Note: Uses self:: prefix because this function shadows the UnaryOp enum
        #[allow(non_snake_case)]
        pub fn UnaryOp(op: self::UnaryOp, expr: Box<TypedExpression>) -> Expression {
            Expression::UnaryOp(op, expr)
        }

        /// Construct Expression::BinaryOp variant
        /// Note: Uses self:: prefix because this function shadows the BinaryOp enum
        #[allow(non_snake_case)]
        pub fn BinaryOp(op: self::BinaryOp, left: Box<TypedExpression>, right: Box<TypedExpression>) -> Expression {
            Expression::BinaryOp(op, left, right)
        }

        /// Construct Expression::IntLiteral variant
        #[allow(non_snake_case)]
        pub fn IntLiteral(n: i64) -> Expression {
            Expression::IntLiteral(n)
        }

        /// Construct Expression::FloatLiteral variant
        #[allow(non_snake_case)]
        pub fn FloatLiteral(n: f64) -> Expression {
            Expression::FloatLiteral(n)
        }

        /// Construct Expression::StringLiteral variant
        #[allow(non_snake_case)]
        pub fn StringLiteral(s: String) -> Expression {
            Expression::StringLiteral(s)
        }

        /// Construct Expression::Identifier variant
        #[allow(non_snake_case)]
        pub fn Identifier(name: String) -> Expression {
            Expression::Identifier(name)
        }
    })
}

/// Format Rust code using rustfmt
/// Falls back to unformatted code if rustfmt is not available
pub fn format_rust_code(code: &str) -> String {
    // Try to run rustfmt
    let result = Command::new("rustfmt")
        .arg("--edition")
        .arg("2021")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    match result {
        Ok(mut child) => {
            // Write code to stdin
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(code.as_bytes());
            }

            // Read formatted output
            match child.wait_with_output() {
                Ok(output) if output.status.success() => {
                    String::from_utf8(output.stdout).unwrap_or_else(|_| code.to_string())
                }
                _ => code.to_string(),
            }
        }
        Err(_) => {
            // rustfmt not available, return unformatted code
            code.to_string()
        }
    }
}

/// Generate a pest-compatible grammar from ZynGrammar rules (public wrapper)
pub fn generate_pest_grammar_string(grammar: &ZynGrammar) -> Result<String> {
    generate_pest_grammar(grammar)
}

/// Strip name bindings from a pattern for pest compatibility.
/// Converts `name:rule` to just `rule` since pest doesn't support named bindings.
///
/// Examples:
/// - `items:top_level_items` -> `top_level_items`
/// - `name:identifier ~ ":" ~ value:expr` -> `identifier ~ ":" ~ expr`
fn strip_name_bindings(pattern: &str) -> String {
    use regex::Regex;
    // Match identifier followed by colon followed by identifier (the binding syntax)
    // We need to handle cases like `name:identifier` but not break strings like `":"`
    let re = Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();
    re.replace_all(pattern, "$2").to_string()
}

/// Generate a pest-compatible grammar from ZynGrammar rules
fn generate_pest_grammar(grammar: &ZynGrammar) -> Result<String> {
    let mut lines = Vec::new();

    // Add header comment
    lines.push(format!(
        "// Generated by ZynPEG from {}.zyn",
        grammar.language.name.to_lowercase()
    ));
    lines.push(String::new());

    // Generate each rule
    for rule in &grammar.rules {
        let modifier = match rule.modifier {
            Some(RuleModifier::Atomic) => "@",
            Some(RuleModifier::Silent) => "_",
            Some(RuleModifier::Compound) => "$",
            Some(RuleModifier::NonAtomic) => "!",
            None => "",
        };

        // Strip name bindings from the pattern for pest compatibility
        let pest_pattern = strip_name_bindings(&rule.pattern);

        lines.push(format!(
            "{} = {}{{ {} }}",
            rule.name, modifier, pest_pattern
        ));
    }

    // Add standard whitespace/comment rules if not defined
    let has_whitespace = grammar.rules.iter().any(|r| r.name == "WHITESPACE");
    let has_comment = grammar.rules.iter().any(|r| r.name == "COMMENT");

    if !has_whitespace {
        lines.push(String::new());
        lines.push("WHITESPACE = _{ \" \" | \"\\t\" | \"\\n\" | \"\\r\" }".to_string());
    }

    if !has_comment {
        lines.push("COMMENT = _{ \"//\" ~ (!\"\\n\" ~ ANY)* ~ \"\\n\"? }".to_string());
    }

    Ok(lines.join("\n"))
}

/// Generate the TypedAST builder code
fn generate_ast_builder(grammar: &ZynGrammar) -> Result<TokenStream> {
    let imports = parse_imports(&grammar.imports.code);
    let context_fields = generate_context_fields(&grammar.context);
    let type_helpers = parse_type_helpers(&grammar.type_helpers.code);

    // Generate build methods for each rule with an action
    let build_methods: Vec<TokenStream> = grammar
        .rules
        .iter()
        .filter(|r| r.action.is_some())
        .map(|r| generate_build_method(r, grammar))
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        // Generated imports
        #imports

        /// TypedAST builder context
        pub struct AstBuilderContext<'a> {
            #context_fields
        }

        impl<'a> AstBuilderContext<'a> {
            #type_helpers

            #(#build_methods)*
        }
    })
}

/// Generate the parser implementation
fn generate_parser_impl(grammar: &ZynGrammar) -> Result<TokenStream> {
    let parser_name = format_ident!("{}Parser", to_pascal_case(&grammar.language.name));
    let grammar_file = format!("{}.pest", grammar.language.name.to_lowercase());

    Ok(quote! {
        use pest_derive::Parser;

        #[derive(Parser)]
        #[grammar = #grammar_file]
        pub struct #parser_name;

        impl #parser_name {
            /// Parse source code to TypedAST
            pub fn parse_to_typed_ast(
                input: &str,
                arena: &mut AstArena,
                type_registry: &mut TypeRegistry,
            ) -> Result<TypedProgram, ParseError> {
                use pest::Parser;

                // Parse with pest
                let pairs = Self::parse(Rule::program, input)?;

                // Build TypedAST
                let mut ctx = AstBuilderContext { arena, type_registry };
                ctx.build_program(pairs)
            }
        }
    })
}

/// Generate standalone AST builder (no external dependencies)
///
/// This version doesn't include the grammar's @imports - instead it expects
/// types to be provided by the typed_ast module (scaffolding).
fn generate_standalone_ast_builder(grammar: &ZynGrammar) -> Result<TokenStream> {
    let type_helpers = parse_type_helpers(&grammar.type_helpers.code);

    // Generate build methods for rules WITH actions
    let action_methods: Vec<TokenStream> = grammar
        .rules
        .iter()
        .filter(|r| r.action.is_some())
        .map(|r| generate_build_method(r, grammar))
        .collect::<Result<Vec<_>>>()?;

    // Generate dispatch methods for rules WITHOUT actions (like declaration, type_expr, etc.)
    let dispatch_methods: Vec<TokenStream> = grammar
        .rules
        .iter()
        .filter(|r| r.action.is_none())
        .filter(|r| !r.name.starts_with("WHITESPACE") && !r.name.starts_with("COMMENT"))
        .filter(|r| {
            r.modifier != Some(RuleModifier::Silent) && r.modifier != Some(RuleModifier::Atomic)
        })
        .map(|r| generate_dispatch_method(r, grammar))
        .collect();

    // Standalone: no external imports, uses generated typed_ast types
    Ok(quote! {
        //! Generated AST builder
        //!
        //! This file is auto-generated by ZynPEG. Do not edit manually.

        #![allow(dead_code, unused_variables, unused_imports, unused_mut)]

        use super::typed_ast::*;
        use super::Rule;  // Import the Rule enum from the parser

        // Type helper for folding postfix operations
        fn fold_postfix_ops(base: TypedExpression, ops: Vec<PostfixOp>) -> TypedExpression {
            ops.into_iter().fold(base, |expr, op| {
                let span = expr.span;
                match op {
                    PostfixOp::Deref => TypedExpression {
                        ty: expr.ty.deref(),
                        expr: Expression::Deref(Box::new(expr)),
                        span,
                    },
                    PostfixOp::Field(name) => TypedExpression {
                        ty: Type::Unknown,
                        expr: Expression::FieldAccess(Box::new(expr), name),
                        span,
                    },
                    PostfixOp::Index(index) => TypedExpression {
                        ty: expr.ty.element_type(),
                        expr: Expression::Index(Box::new(expr), Box::new(index)),
                        span,
                    },
                    PostfixOp::Call(args) => TypedExpression {
                        ty: expr.ty.return_type(),
                        expr: Expression::Call(Box::new(expr), args),
                        span,
                    },
                    PostfixOp::OptionalUnwrap => TypedExpression {
                        ty: expr.ty.unwrap_optional(),
                        expr: Expression::UnaryOp(UnaryOp::Try, Box::new(expr)),
                        span,
                    },
                    PostfixOp::TryUnwrap => TypedExpression {
                        ty: expr.ty.unwrap_result(),
                        expr: Expression::UnaryOp(UnaryOp::Try, Box::new(expr)),
                        span,
                    },
                }
            })
        }

        /// TypedAST builder context
        pub struct AstBuilderContext;

        impl AstBuilderContext {
            pub fn new() -> Self {
                Self
            }

            #type_helpers

            #(#action_methods)*

            #(#dispatch_methods)*
        }
    })
}

/// Generate a dispatch method for a rule without an action
/// These methods dispatch to child rules based on match
fn generate_dispatch_method(rule: &RuleDef, grammar: &ZynGrammar) -> TokenStream {
    let method_name = format_ident!("build_{}", rule.name);

    // Parse the pattern to find child rules
    let child_rules = extract_child_rules(&rule.pattern);

    // Filter to rules that exist in grammar AND have build methods
    // (i.e., rules with actions OR non-atomic rules without actions that get dispatch methods)
    let dispatchable: Vec<_> = child_rules.iter()
        .filter(|name| {
            grammar.rules.iter().any(|r| {
                &r.name == *name && (
                    // Has an action - will have a build method
                    r.action.is_some() ||
                    // Non-atomic rule without action - will get a dispatch method
                    (r.modifier != Some(RuleModifier::Atomic) && r.modifier != Some(RuleModifier::Silent))
                )
            })
        })
        .collect();

    // Determine return type based on first child with an action
    let return_type_str = dispatchable
        .iter()
        .find_map(|name| {
            grammar
                .rules
                .iter()
                .find(|r| &r.name == *name)
                .and_then(|r| r.action.as_ref())
                .map(|a| a.return_type.clone())
        })
        .unwrap_or_else(|| "TypedNode<TypedExpression>".to_string());

    // Filter dispatchable to only include rules with compatible return types
    // Rules with actions must have a return type that matches the dispatch method's return type
    let dispatchable: Vec<_> = dispatchable
        .into_iter()
        .filter(|name| {
            if let Some(rule) = grammar.rules.iter().find(|r| &r.name == *name) {
                if let Some(action) = &rule.action {
                    // If the action returns a different type (e.g., String vs TypedNode<TypedExpression>),
                    // exclude it from the dispatch
                    let child_return = &action.return_type;
                    // Check if return types are compatible
                    // Simple types like String, bool, i64 are incompatible with TypedNode<...>
                    let is_simple_type = child_return == "String"
                        || child_return == "bool"
                        || child_return == "i64"
                        || child_return == "i32"
                        || child_return == "f64"
                        || child_return == "f32";
                    let expected_is_typed_node = return_type_str.contains("TypedNode")
                        || return_type_str.contains("TypedExpression")
                        || return_type_str.contains("TypedDeclaration")
                        || return_type_str.contains("TypedStatement")
                        || return_type_str == "Type";

                    // Exclude if child returns simple type but we expect complex type
                    if is_simple_type && expected_is_typed_node {
                        return false;
                    }
                    // Exclude if return types are explicitly different
                    if child_return != &return_type_str
                        && !return_type_str.contains(child_return)
                        && !child_return.contains(&return_type_str)
                    {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    let return_type: TokenStream = return_type_str
        .parse()
        .unwrap_or_else(|_| quote! { TypedNode<TypedExpression> });

    if dispatchable.is_empty() {
        // No child rules to dispatch to - return a default/passthrough
        quote! {
            pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<TypedExpression, ParseError> {
                let span = Span::from_pest(pair.as_span());
                let text = pair.as_str().trim();

                // Try to interpret as literal or identifier
                if let Ok(n) = text.parse::<i64>() {
                    return Ok(TypedExpression {
                        expr: Expression::IntLiteral(n),
                        ty: Type::I64,
                        span,
                    });
                }
                if let Ok(n) = text.parse::<f64>() {
                    return Ok(TypedExpression {
                        expr: Expression::FloatLiteral(n),
                        ty: Type::F64,
                        span,
                    });
                }
                if text == "true" {
                    return Ok(TypedExpression {
                        expr: Expression::BoolLiteral(true),
                        ty: Type::Bool,
                        span,
                    });
                }
                if text == "false" {
                    return Ok(TypedExpression {
                        expr: Expression::BoolLiteral(false),
                        ty: Type::Bool,
                        span,
                    });
                }

                Ok(TypedExpression {
                    expr: Expression::Identifier(text.to_string()),
                    ty: Type::Unknown,
                    span,
                })
            }
        }
    } else {
        // Generate match arms for child rules
        let match_arms: Vec<TokenStream> = dispatchable
            .iter()
            .map(|child_name| {
                let rule_variant = format_ident!("{}", child_name);
                let build_method = format_ident!("build_{}", child_name);
                quote! {
                    Rule::#rule_variant => self.#build_method(inner),
                }
            })
            .collect();

        quote! {
            pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<#return_type, ParseError> {
                let span = Span::from_pest(pair.as_span());
                if let Some(inner) = pair.into_inner().next() {
                    match inner.as_rule() {
                        #(#match_arms)*
                        _ => Err(ParseError(format!("Unexpected rule in {}: {:?}", stringify!(#method_name), inner.as_rule()))),
                    }
                } else {
                    Err(ParseError(format!("Empty {} rule", stringify!(#method_name))))
                }
            }
        }
    }
}

/// Extract child rule names from a pattern
fn extract_child_rules(pattern: &str) -> Vec<String> {
    let mut rules = Vec::new();

    // Simple extraction - find lowercase identifiers that aren't keywords
    let keywords = [
        "SOI",
        "EOI",
        "ANY",
        "ASCII",
        "ASCII_DIGIT",
        "ASCII_ALPHA",
        "ASCII_ALPHANUMERIC",
        "WHITESPACE",
        "COMMENT",
    ];

    for part in pattern.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let part = part.trim();
        if !part.is_empty()
            && part
                .chars()
                .next()
                .map(|c| c.is_lowercase())
                .unwrap_or(false)
            && !keywords.contains(&part)
        {
            if !rules.contains(&part.to_string()) {
                rules.push(part.to_string());
            }
        }
    }

    rules
}

/// Generate standalone parser implementation (simpler, no external types)
fn generate_standalone_parser_impl(_grammar: &ZynGrammar) -> Result<TokenStream> {
    Ok(quote! {
        //! Generated parser implementation
        //!
        //! This file is auto-generated by ZynPEG. Do not edit manually.

        #![allow(unused_imports)]

        // Note: The actual parser struct is defined in lib.rs using pest_derive
        // This file just provides the parse_to_typed_ast helper.

        use super::typed_ast::*;
        use super::ast_builder::AstBuilderContext;
        use super::Rule;
        use pest::Parser;

        /// Parse source code to TypedProgram
        pub fn parse_to_typed_ast<P: pest::Parser<Rule>>(
            input: &str,
        ) -> Result<TypedProgram, ParseError> {
            // Parse with pest
            let pairs = P::parse(Rule::program, input)?;

            // Build TypedAST
            let mut ctx = AstBuilderContext::new();
            for pair in pairs {
                if pair.as_rule() == Rule::program {
                    return ctx.build_program(pair);
                }
            }

            Err(ParseError("No program rule found".to_string()))
        }
    })
}

// ============================================================================
// ZYNTAX-COMPATIBLE PARSER GENERATION
// ============================================================================
// These functions generate code that uses zyntax_typed_ast types directly,
// enabling JIT compilation via zyntax_compiler.

/// Generate AST builder that uses zyntax_typed_ast types
fn generate_zyntax_ast_builder(grammar: &ZynGrammar) -> Result<TokenStream> {
    let type_helpers = parse_type_helpers(&grammar.type_helpers.code);

    // Generate build methods for rules WITH actions
    let action_methods: Vec<TokenStream> = grammar
        .rules
        .iter()
        .filter(|r| r.action.is_some())
        .map(|r| generate_zyntax_build_method(r, grammar))
        .collect::<Result<Vec<_>>>()?;

    // Generate dispatch methods for rules WITHOUT actions
    let dispatch_methods: Vec<TokenStream> = grammar
        .rules
        .iter()
        .filter(|r| r.action.is_none())
        .filter(|r| !r.name.starts_with("WHITESPACE") && !r.name.starts_with("COMMENT"))
        .filter(|r| {
            r.modifier != Some(RuleModifier::Silent) && r.modifier != Some(RuleModifier::Atomic)
        })
        .map(|r| generate_zyntax_dispatch_method(r, grammar))
        .collect();

    Ok(quote! {
        //! Generated AST builder for zyntax_typed_ast
        //!
        //! This file is auto-generated by ZynPEG. Do not edit manually.
        //! Uses zyntax_typed_ast types for JIT compilation compatibility.

        #![allow(dead_code, unused_variables, unused_imports, unused_mut)]

        use zyntax_typed_ast::{
            Type, PrimitiveType, TypedNode, TypedProgram, TypedDeclaration,
            TypedFunction, TypedVariable, TypedStatement, TypedExpression,
            TypedLiteral, TypedBlock, TypedLet, TypedIf, TypedWhile, TypedFor,
            TypedBinary, TypedUnary, TypedCall, TypedParameter, TypedMatch, TypedMatchExpr, TypedMatchArm,
            TypedFieldAccess, TypedIndex, TypedLambda, TypedLambdaBody, TypedMethodCall, TypedRange,
            TypedStructLiteral, TypedFieldInit, TypedPattern, TypedLiteralPattern,
            TypedReference, TypedCast, TypedIfExpr,
            TypedDefer, TypedClass, TypedEnum, TypedField, TypedVariant, TypedTypeParam,
            BinaryOp, UnaryOp, Span, InternedString, Mutability, Visibility,
            CallingConvention, ParameterKind, typed_node, NullabilityKind, AsyncKind,
            TypeId, ConstValue, Variance,
        };
        use super::Rule;

        /// Postfix operation for expression building
        #[derive(Debug, Clone)]
        pub enum PostfixOp {
            Deref,
            Field(InternedString),
            Index(TypedNode<TypedExpression>),
            Call(Vec<TypedNode<TypedExpression>>),
            OptionalUnwrap,
            TryUnwrap,
        }

        /// Parse error type
        #[derive(Debug)]
        pub struct ParseError(pub String);

        impl std::fmt::Display for ParseError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl std::error::Error for ParseError {}

        impl<R: pest::RuleType> From<pest::error::Error<R>> for ParseError {
            fn from(e: pest::error::Error<R>) -> Self {
                ParseError(e.to_string())
            }
        }

        /// Helper: Create Span from pest span
        fn span_from_pest(pest_span: pest::Span) -> Span {
            Span::new(pest_span.start(), pest_span.end())
        }

        /// Helper: Intern a string using the global interner
        fn intern(s: &str) -> InternedString {
            InternedString::new_global(s)
        }

        /// Helper: Parse primitive type from string
        fn parse_primitive_type(s: &str) -> Type {
            match s {
                "i8" => Type::Primitive(PrimitiveType::I8),
                "i16" => Type::Primitive(PrimitiveType::I16),
                "i32" => Type::Primitive(PrimitiveType::I32),
                "i64" => Type::Primitive(PrimitiveType::I64),
                "i128" => Type::Primitive(PrimitiveType::I128),
                "u8" => Type::Primitive(PrimitiveType::U8),
                "u16" => Type::Primitive(PrimitiveType::U16),
                "u32" => Type::Primitive(PrimitiveType::U32),
                "u64" => Type::Primitive(PrimitiveType::U64),
                "u128" => Type::Primitive(PrimitiveType::U128),
                "f32" => Type::Primitive(PrimitiveType::F32),
                "f64" => Type::Primitive(PrimitiveType::F64),
                "bool" => Type::Primitive(PrimitiveType::Bool),
                "char" => Type::Primitive(PrimitiveType::Char),
                "void" | "()" => Type::Primitive(PrimitiveType::Unit),
                "usize" => Type::Primitive(PrimitiveType::USize),
                "isize" => Type::Primitive(PrimitiveType::ISize),
                _ => Type::Never, // Unknown type
            }
        }

        /// TypedAST builder context
        pub struct AstBuilderContext;

        impl AstBuilderContext {
            pub fn new() -> Self {
                Self
            }

            #type_helpers

            #(#action_methods)*

            #(#dispatch_methods)*
        }
    })
}

/// Generate parser implementation that uses zyntax_typed_ast
fn generate_zyntax_parser_impl(_grammar: &ZynGrammar) -> Result<TokenStream> {
    Ok(quote! {
        //! Generated parser implementation for zyntax_typed_ast
        //!
        //! This file is auto-generated by ZynPEG. Do not edit manually.

        #![allow(unused_imports)]

        use zyntax_typed_ast::TypedProgram;
        use super::ast_builder::{AstBuilderContext, ParseError};
        use super::Rule;
        use pest::Parser;

        /// Parse source code to TypedProgram (zyntax_typed_ast compatible)
        pub fn parse_to_typed_ast<P: pest::Parser<Rule>>(
            input: &str,
        ) -> Result<TypedProgram, ParseError> {
            let pairs = P::parse(Rule::program, input)?;

            let mut ctx = AstBuilderContext::new();
            for pair in pairs {
                if pair.as_rule() == Rule::program {
                    return ctx.build_program(pair);
                }
            }

            Err(ParseError("No program rule found".to_string()))
        }
    })
}

/// Generate a build method for a rule using zyntax_typed_ast types
fn generate_zyntax_build_method(rule: &RuleDef, _grammar: &ZynGrammar) -> Result<TokenStream> {
    let action = rule
        .action
        .as_ref()
        .ok_or_else(|| ZynPegError::InvalidAction("No action for rule".into()))?;

    let method_name = format_ident!("build_{}", rule.name);

    // Map return types to zyntax_typed_ast types
    let return_type_str = map_to_zyntax_type(&action.return_type);
    // Parse as syn::Type first to properly handle generics, then convert to TokenStream
    let return_type: TokenStream = syn::parse_str::<syn::Type>(&return_type_str)
        .map(|t| quote! { #t })
        .unwrap_or_else(|_| return_type_str.parse().unwrap_or_else(|_| quote! { () }));

    let pattern_info = analyze_pattern(&rule.pattern);
    let child_extraction = generate_child_extraction(&pattern_info);

    // Check if we need to wrap with typed_node()
    let needs_typed_node_wrap = needs_typed_node_wrapper(&action.return_type);

    let body = if let Some(ref raw_code) = action.raw_code {
        let code = transform_zyntax_captures(raw_code, &pattern_info);
        // Parse as syn::Block or syn::Expr to get proper token spacing
        let code_tokens: TokenStream = if code.trim().starts_with('{') {
            syn::parse_str::<syn::Block>(&code)
                .map(|b| quote! { #b })
                .unwrap_or_else(|_| {
                    // Fallback: try as expression
                    syn::parse_str::<syn::Expr>(&code)
                        .map(|e| quote! { { #e } })
                        .unwrap_or_else(|_| quote! { todo!("Failed to parse raw code") })
                })
        } else {
            // Try parsing as expression first
            syn::parse_str::<syn::Expr>(&code)
                .map(|e| quote! { { #e } })
                .unwrap_or_else(|_| {
                    // Fallback: try as statement(s) wrapped in block
                    syn::parse_str::<syn::Block>(&format!("{{ {} }}", code))
                        .map(|b| quote! { #b })
                        .unwrap_or_else(|_| quote! { todo!("Failed to parse raw code") })
                })
        };
        code_tokens
    } else {
        // For structured field actions, we need to build the appropriate type
        generate_zyntax_struct_body(action, &pattern_info, needs_typed_node_wrap)
    };

    // Parse child extraction as a block of statements
    let child_extraction_tokens: TokenStream = if child_extraction.trim().is_empty() {
        quote! {}
    } else {
        syn::parse_str::<syn::Block>(&format!("{{ {} }}", child_extraction))
            .map(|b| {
                // Extract inner statements without the braces
                let stmts = &b.stmts;
                quote! { #(#stmts)* }
            })
            .unwrap_or_else(|_| child_extraction.parse().unwrap_or_else(|_| quote! {}))
    };

    Ok(quote! {
        pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<#return_type, ParseError> {
            let span = span_from_pest(pair.as_span());
            let pair_str = pair.as_str();
            let mut children = pair.into_inner().peekable();

            #child_extraction_tokens

            Ok(#body)
        }
    })
}

/// Check if a return type needs typed_node() wrapping
fn needs_typed_node_wrapper(type_str: &str) -> bool {
    matches!(
        type_str,
        "TypedDeclaration" | "TypedStatement" | "TypedExpression"
    )
}

/// Generate struct body for zyntax_typed_ast, handling typed_node wrapping
fn generate_zyntax_struct_body(
    action: &ActionBlock,
    pattern_info: &[PatternElement],
    needs_wrap: bool,
) -> TokenStream {
    // Check if this is a TypedDeclaration with "decl" field - needs special handling
    let has_decl_field = action.fields.iter().any(|f| f.name == "decl");
    let has_stmt_field = action.fields.iter().any(|f| f.name == "stmt");
    let has_expr_field = action
        .fields
        .iter()
        .any(|f| f.name == "expr" && action.return_type.contains("Expression"));

    if needs_wrap && has_decl_field {
        // TypedDeclaration with decl field - extract the inner declaration and wrap
        let decl_field = action.fields.iter().find(|f| f.name == "decl").unwrap();
        let decl_value = transform_zyntax_captures(&decl_field.value, pattern_info);

        // Parse and convert the declaration value to proper enum variant
        let inner_expr = convert_declaration_to_zyntax(&decl_value, pattern_info);

        quote! {
            typed_node(#inner_expr, Type::Never, span)
        }
    } else if needs_wrap && has_stmt_field {
        // TypedStatement with stmt field - convert to enum variant
        let stmt_field = action.fields.iter().find(|f| f.name == "stmt").unwrap();
        let stmt_value = transform_zyntax_captures(&stmt_field.value, pattern_info);

        let inner_expr = convert_statement_to_zyntax(&stmt_value, pattern_info);

        quote! {
            typed_node(#inner_expr, Type::Never, span)
        }
    } else if needs_wrap && has_expr_field {
        // TypedExpression with expr field - convert to enum variant
        let expr_field = action.fields.iter().find(|f| f.name == "expr").unwrap();
        let expr_value = transform_zyntax_captures(&expr_field.value, pattern_info);
        let ty_field = action.fields.iter().find(|f| f.name == "ty");

        let inner_expr = convert_expression_to_zyntax(&expr_value, pattern_info);
        let ty_expr = if let Some(ty) = ty_field {
            let ty_value = transform_zyntax_captures(&ty.value, pattern_info);
            syn::parse_str::<syn::Expr>(&ty_value)
                .map(|e| quote! { #e })
                .unwrap_or_else(|_| quote! { Type::Never })
        } else {
            quote! { Type::Never }
        };

        quote! {
            typed_node(#inner_expr, #ty_expr, span)
        }
    } else {
        // Regular struct - build normally
        let field_assignments: Vec<TokenStream> = action
            .fields
            .iter()
            .map(|f| {
                let name = format_ident!("{}", f.name);
                let value = transform_zyntax_captures(&f.value, pattern_info);
                let value_tokens: TokenStream = syn::parse_str::<syn::Expr>(&value)
                    .map(|e| quote! { #e })
                    .unwrap_or_else(|_| quote! { todo!("Failed to parse field value") });
                quote! { #name: #value_tokens }
            })
            .collect();

        let return_type_str = map_to_zyntax_type(&action.return_type);
        let return_type_mapped: TokenStream = syn::parse_str::<syn::Type>(&return_type_str)
            .map(|t| quote! { #t })
            .unwrap_or_else(|_| return_type_str.parse().unwrap_or_else(|_| quote! { () }));

        if field_assignments.is_empty() {
            quote! {
                <#return_type_mapped>::default()
            }
        } else {
            let result: TokenStream = quote! {
                #return_type_mapped {
                    #(#field_assignments,)*
                }
            };
            result
        }
    }
}

/// Convert a Declaration::* construct to TypedDeclaration::* for zyntax_typed_ast
fn convert_declaration_to_zyntax(decl_str: &str, _pattern_info: &[PatternElement]) -> TokenStream {
    // The grammar uses patterns like "Declaration::Const(ConstDecl { ... })"
    // We need to convert to "TypedDeclaration::Variable(TypedVariable { ... })"

    // For now, we'll parse and transform the string
    // This is a simplistic transformation - a full solution would properly parse the AST

    let transformed = decl_str
        .replace("Declaration::Const(ConstDecl", "TypedDeclaration::Variable(TypedVariable")
        .replace("Declaration::Var(VarDecl", "TypedDeclaration::Variable(TypedVariable")
        .replace("Declaration::Function(FnDecl", "TypedDeclaration::Function(TypedFunction")
        .replace("Declaration::Struct(StructDecl", "TypedDeclaration::Class(TypedClass")  // Map to Class for now
        .replace("Declaration::Enum(EnumDecl", "TypedDeclaration::Enum(TypedEnum")
        .replace("Declaration::Union(UnionDecl", "TypedDeclaration::Class(TypedClass")  // Map union to class
        .replace("Declaration::ErrorSet(ErrorSetDecl", "TypedDeclaration::Enum(TypedEnum")  // Map to enum
        // Field name mappings
        .replace("is_pub:", "visibility: if is_pub { Visibility::Public } else { Visibility::Private },\n        //")
        .replace("name: intern(", "name: InternedString::new_global(")
        .replace("value:", "initializer: Some(Box::new(")
        .replace("is_export:", "is_external:")
        .replace("is_extern:", "is_external:");

    syn::parse_str::<syn::Expr>(&transformed)
        .map(|e| quote! { #e })
        .unwrap_or_else(|_| {
            // Fallback: return a placeholder
            quote! {
                TypedDeclaration::Variable(TypedVariable {
                    name: InternedString::new_global("unknown"),
                    ty: Type::Never,
                    mutability: Mutability::Immutable,
                    initializer: None,
                    visibility: Visibility::Private,
                })
            }
        })
}

/// Convert a Statement::* construct to TypedStatement::* for zyntax_typed_ast
fn convert_statement_to_zyntax(stmt_str: &str, _pattern_info: &[PatternElement]) -> TokenStream {
    // Transform statement patterns
    let transformed = stmt_str
        .replace("Statement::Const(", "TypedStatement::Let(TypedLet { name: ")
        .replace("Statement::Let(", "TypedStatement::Let(TypedLet { name: ")
        .replace("Statement::Return(", "TypedStatement::Return(")
        .replace("Statement::If(", "TypedStatement::If(TypedIf { condition: Box::new(")
        .replace("Statement::While(", "TypedStatement::While(TypedWhile { condition: Box::new(")
        .replace("Statement::For(", "TypedStatement::For(TypedFor { ")
        .replace("Statement::Break(", "TypedStatement::Break(")
        .replace("Statement::Continue", "TypedStatement::Continue")
        .replace("Statement::Expression(", "TypedStatement::Expression(Box::new(")
        .replace("Statement::Block(", "TypedStatement::Block(TypedBlock { statements: ")
        .replace("Statement::Defer(", "TypedStatement::Defer(TypedDefer { body: ")
        .replace("Statement::Assign(", "TypedStatement::Expression(Box::new(typed_node(TypedExpression::Binary(TypedBinary { op: BinaryOp::Assign, left: Box::new(");

    syn::parse_str::<syn::Expr>(&transformed)
        .map(|e| quote! { #e })
        .unwrap_or_else(|_| {
            quote! { TypedStatement::Continue }
        })
}

/// Convert an Expression::* construct to TypedExpression::* for zyntax_typed_ast
fn convert_expression_to_zyntax(expr_str: &str, _pattern_info: &[PatternElement]) -> TokenStream {
    let transformed = expr_str
        .replace(
            "Expression::IntLiteral(",
            "TypedExpression::Literal(TypedLiteral::Integer(",
        )
        .replace(
            "Expression::FloatLiteral(",
            "TypedExpression::Literal(TypedLiteral::Float(",
        )
        .replace(
            "Expression::BoolLiteral(",
            "TypedExpression::Literal(TypedLiteral::Bool(",
        )
        .replace(
            "Expression::StringLiteral(",
            "TypedExpression::Literal(TypedLiteral::String(InternedString::new_global(",
        )
        .replace(
            "Expression::NullLiteral",
            "TypedExpression::Literal(TypedLiteral::Null)",
        )
        .replace(
            "Expression::UndefinedLiteral",
            "TypedExpression::Literal(TypedLiteral::Undefined)",
        )
        .replace(
            "Expression::BinaryOp(",
            "TypedExpression::Binary(TypedBinary { op: ",
        )
        .replace(
            "Expression::UnaryOp(",
            "TypedExpression::Unary(TypedUnary { op: ",
        )
        .replace(
            "Expression::Call(",
            "TypedExpression::Call(TypedCall { callee: ",
        )
        .replace(
            "Expression::FieldAccess(",
            "TypedExpression::Field(TypedFieldAccess { object: ",
        )
        .replace(
            "Expression::Index(",
            "TypedExpression::Index(TypedIndex { array: ",
        )
        .replace("Expression::Deref(", "TypedExpression::Dereference(")
        .replace("Expression::Array(", "TypedExpression::Array(")
        .replace(
            "Expression::Lambda(",
            "TypedExpression::Lambda(TypedLambda ",
        )
        .replace("Expression::Try(", "TypedExpression::Try(")
        .replace(
            "Expression::Switch(",
            "TypedExpression::Match(TypedMatchExpr ",
        )
        .replace(
            "Expression::StructLiteral(",
            "TypedExpression::Struct(TypedStructLiteral ",
        );

    syn::parse_str::<syn::Expr>(&transformed)
        .map(|e| quote! { #e })
        .unwrap_or_else(|_| {
            quote! { TypedExpression::Literal(TypedLiteral::Unit) }
        })
}

/// Generate a dispatch method using zyntax_typed_ast types
fn generate_zyntax_dispatch_method(rule: &RuleDef, grammar: &ZynGrammar) -> TokenStream {
    let method_name = format_ident!("build_{}", rule.name);
    let child_rules = extract_child_rules(&rule.pattern);

    let dispatchable: Vec<_> = child_rules
        .iter()
        .filter(|name| {
            grammar.rules.iter().any(|r| {
                &r.name == *name
                    && (r.action.is_some()
                        || (r.modifier != Some(RuleModifier::Atomic)
                            && r.modifier != Some(RuleModifier::Silent)))
            })
        })
        .collect();

    // Determine return type
    let return_type_str = dispatchable
        .iter()
        .find_map(|name| {
            grammar
                .rules
                .iter()
                .find(|r| &r.name == *name)
                .and_then(|r| r.action.as_ref())
                .map(|a| map_to_zyntax_type(&a.return_type))
        })
        .unwrap_or_else(|| "TypedNode<TypedExpression>".to_string());

    let return_type: TokenStream = syn::parse_str::<syn::Type>(&return_type_str)
        .map(|t| quote! { #t })
        .unwrap_or_else(|_| quote! { TypedNode<TypedExpression> });

    if dispatchable.is_empty() {
        quote! {
            pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<TypedNode<TypedExpression>, ParseError> {
                let span = span_from_pest(pair.as_span());
                let text = pair.as_str().trim();

                // Try to interpret as literal
                if let Ok(n) = text.parse::<i128>() {
                    return Ok(typed_node(
                        TypedExpression::Literal(TypedLiteral::Integer(n)),
                        Type::Primitive(PrimitiveType::I64),
                        span,
                    ));
                }
                if let Ok(n) = text.parse::<f64>() {
                    return Ok(typed_node(
                        TypedExpression::Literal(TypedLiteral::Float(n)),
                        Type::Primitive(PrimitiveType::F64),
                        span,
                    ));
                }
                if text == "true" {
                    return Ok(typed_node(
                        TypedExpression::Literal(TypedLiteral::Bool(true)),
                        Type::Primitive(PrimitiveType::Bool),
                        span,
                    ));
                }
                if text == "false" {
                    return Ok(typed_node(
                        TypedExpression::Literal(TypedLiteral::Bool(false)),
                        Type::Primitive(PrimitiveType::Bool),
                        span,
                    ));
                }

                Ok(typed_node(
                    TypedExpression::Variable(intern(text)),
                    Type::Never,
                    span,
                ))
            }
        }
    } else {
        let match_arms: Vec<TokenStream> = dispatchable
            .iter()
            .map(|child_name| {
                let rule_variant = format_ident!("{}", child_name);
                let build_method = format_ident!("build_{}", child_name);
                quote! {
                    Rule::#rule_variant => self.#build_method(inner),
                }
            })
            .collect();

        quote! {
            pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<#return_type, ParseError> {
                let span = span_from_pest(pair.as_span());
                if let Some(inner) = pair.into_inner().next() {
                    match inner.as_rule() {
                        #(#match_arms)*
                        _ => Err(ParseError(format!("Unexpected rule in {}: {:?}", stringify!(#method_name), inner.as_rule()))),
                    }
                } else {
                    Err(ParseError(format!("Empty {} rule", stringify!(#method_name))))
                }
            }
        }
    }
}

/// Map standalone type names to zyntax_typed_ast equivalents
fn map_to_zyntax_type(type_str: &str) -> String {
    match type_str {
        "TypedExpression" => "TypedNode<TypedExpression>".to_string(),
        "TypedStatement" => "TypedNode<TypedStatement>".to_string(),
        "TypedDeclaration" => "TypedNode<TypedDeclaration>".to_string(),
        "TypedProgram" => "TypedProgram".to_string(),
        "TypedBlock" => "TypedBlock".to_string(),
        "Type" => "Type".to_string(),
        "Param" | "TypedParam" => "TypedParameter".to_string(),
        "Vec<Param>" | "Vec<TypedParam>" => "Vec<TypedParameter>".to_string(),
        "Vec<TypedStatement>" => "Vec<TypedNode<TypedStatement>>".to_string(),
        "Vec<TypedExpression>" => "Vec<TypedNode<TypedExpression>>".to_string(),
        s if s.starts_with("Vec<") => s.to_string(),
        s if s.starts_with("Option<") => s.to_string(),
        _ => type_str.to_string(),
    }
}

/// Transform capture references for zyntax_typed_ast
fn transform_zyntax_captures(value: &str, pattern: &[PatternElement]) -> String {
    let mut result = transform_captures_to_vars(value, pattern);

    // Replace Type::I32 with Type::Primitive(PrimitiveType::I32) etc.
    // BUT only if not already wrapped in Type::Primitive(...)
    // This prevents double-wrapping when the grammar already uses the correct form
    let type_mappings = [
        ("Type::I8", "Type::Primitive(PrimitiveType::I8)"),
        ("Type::I16", "Type::Primitive(PrimitiveType::I16)"),
        ("Type::I32", "Type::Primitive(PrimitiveType::I32)"),
        ("Type::I64", "Type::Primitive(PrimitiveType::I64)"),
        ("Type::I128", "Type::Primitive(PrimitiveType::I128)"),
        ("Type::U8", "Type::Primitive(PrimitiveType::U8)"),
        ("Type::U16", "Type::Primitive(PrimitiveType::U16)"),
        ("Type::U32", "Type::Primitive(PrimitiveType::U32)"),
        ("Type::U64", "Type::Primitive(PrimitiveType::U64)"),
        ("Type::U128", "Type::Primitive(PrimitiveType::U128)"),
        ("Type::F32", "Type::Primitive(PrimitiveType::F32)"),
        ("Type::F64", "Type::Primitive(PrimitiveType::F64)"),
        ("Type::Bool", "Type::Primitive(PrimitiveType::Bool)"),
        ("Type::Void", "Type::Primitive(PrimitiveType::Unit)"),
        ("Type::String", "Type::Primitive(PrimitiveType::String)"),
        ("Type::Char", "Type::Primitive(PrimitiveType::Char)"),
    ];

    for (from, to) in type_mappings {
        // Only replace if it's a standalone Type::X (not already wrapped)
        // Use word boundary detection by checking character before "Type::"
        let mut new_result = String::new();
        let mut remaining = result.as_str();
        while let Some(idx) = remaining.find(from) {
            // Check if preceded by a word character (letter, digit, _, or ::)
            // If so, it's likely already wrapped (e.g., PrimitiveType::Bool)
            let before = &result[..result.len() - remaining.len() + idx];
            let is_part_of_longer_ident = before.ends_with("Primitive") || before.ends_with("::");
            if is_part_of_longer_ident {
                // Skip this match, it's part of a longer identifier
                new_result.push_str(&remaining[..idx + from.len()]);
                remaining = &remaining[idx + from.len()..];
            } else {
                new_result.push_str(&remaining[..idx]);
                new_result.push_str(to);
                remaining = &remaining[idx + from.len()..];
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }

    // Replace intern() with InternedString::new_global()
    result = result.replace("intern(", "InternedString::new_global(");

    result
}

/// Generate a build method for a rule with an action
fn generate_build_method(rule: &RuleDef, _grammar: &ZynGrammar) -> Result<TokenStream> {
    let action = rule
        .action
        .as_ref()
        .ok_or_else(|| ZynPegError::InvalidAction("No action for rule".into()))?;

    let method_name = format_ident!("build_{}", rule.name);
    let return_type: TokenStream = action
        .return_type
        .parse()
        .map_err(|e| ZynPegError::CodeGenError(format!("Invalid return type: {}", e)))?;

    // Analyze the rule pattern to understand what children exist
    let pattern_info = analyze_pattern(&rule.pattern);

    // Generate child extraction code based on pattern analysis
    let child_extraction = generate_child_extraction(&pattern_info);

    // Check if this is a raw code action or structured fields
    let body = if let Some(ref raw_code) = action.raw_code {
        // Raw code action - transform captures to proper variable access
        let code = transform_captures_to_vars(raw_code, &pattern_info);
        // Parse as syn::Block or syn::Expr to get proper token spacing
        let code_tokens: TokenStream = if code.trim().starts_with('{') {
            syn::parse_str::<syn::Block>(&code)
                .map(|b| quote! { #b })
                .unwrap_or_else(|_| {
                    syn::parse_str::<syn::Expr>(&code)
                        .map(|e| quote! { { #e } })
                        .unwrap_or_else(|_| quote! { todo!("Failed to parse raw code") })
                })
        } else {
            syn::parse_str::<syn::Expr>(&code)
                .map(|e| quote! { { #e } })
                .unwrap_or_else(|_| {
                    syn::parse_str::<syn::Block>(&format!("{{ {} }}", code))
                        .map(|b| quote! { #b })
                        .unwrap_or_else(|_| quote! { todo!("Failed to parse raw code") })
                })
        };
        code_tokens
    } else {
        // Structured field assignments
        let field_assignments: Vec<TokenStream> = action
            .fields
            .iter()
            .map(|f| {
                let name = format_ident!("{}", f.name);
                let value = transform_captures_to_vars(&f.value, &pattern_info);
                // Parse field value as syn::Expr for proper spacing
                let value_tokens: TokenStream = syn::parse_str::<syn::Expr>(&value)
                    .map(|e| quote! { #e })
                    .unwrap_or_else(|_| quote! { todo!("Failed to parse field value") });
                quote! { #name: #value_tokens }
            })
            .collect();

        quote! {
            #return_type {
                #(#field_assignments,)*
            }
        }
    };

    // Parse child extraction as a block of statements
    let child_extraction_tokens: TokenStream = if child_extraction.trim().is_empty() {
        quote! {}
    } else {
        syn::parse_str::<syn::Block>(&format!("{{ {} }}", child_extraction))
            .map(|b| {
                let stmts = &b.stmts;
                quote! { #(#stmts)* }
            })
            .unwrap_or_else(|_| child_extraction.parse().unwrap_or_else(|_| quote! {}))
    };

    Ok(quote! {
        /// Build a #return_type from a parsed rule
        pub fn #method_name(&mut self, pair: pest::iterators::Pair<Rule>) -> Result<#return_type, ParseError> {
            let span = Span::from_pest(pair.as_span());
            let pair_str = pair.as_str();  // Capture text before consuming pair
            let mut children = pair.into_inner().peekable();

            #child_extraction_tokens

            Ok(#body)
        }
    })
}

/// Information about a pattern element
#[derive(Debug, Clone)]
struct PatternElement {
    index: usize, // 1-based index in the pattern
    kind: PatternKind,
    name: Option<String>, // Named element (identifier, type_expr, etc.)
    optional: bool,       // Wrapped in (...)? or includes ?
    repeated: bool,       // Wrapped in (...)* or (...)+
}

#[derive(Debug, Clone)]
enum PatternKind {
    Literal(String), // "const", ";", etc.
    Rule(String),    // identifier, expr, type_expr, etc.
    Builtin(String), // ASCII_DIGIT, ANY, etc.
}

/// Split a pattern by top-level alternation `|`, respecting parentheses
///
/// "a ~ b | c" -> ["a ~ b", "c"]
/// "(a | b) ~ c" -> ["(a | b) ~ c"]  (alternation inside parens is not top-level)
fn split_top_level_alternation(pattern: &str) -> Vec<String> {
    let mut branches = Vec::new();
    let mut depth = 0;
    let mut current = String::new();

    for ch in pattern.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            '|' if depth == 0 => {
                if !current.trim().is_empty() {
                    branches.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        branches.push(current.trim().to_string());
    }

    if branches.is_empty() {
        vec![pattern.to_string()]
    } else {
        branches
    }
}

/// Analyze a pattern to understand its structure
///
/// Groups like `(":" ~ type_expr)?` are treated as a single element that captures
/// the rule inside (type_expr). Literals inside groups are ignored for capture purposes.
///
/// For alternation patterns like `a ~ b | c`, we analyze the first branch (`a ~ b`)
/// since action blocks typically only reference elements from one branch at a time.
fn analyze_pattern(pattern: &str) -> Vec<PatternElement> {
    let mut elements = Vec::new();
    let mut index = 1;

    // First, handle top-level alternation - take first branch only
    // Pattern like "unary_op ~ unary | postfix" -> analyze "unary_op ~ unary"
    let branches = split_top_level_alternation(pattern);
    let pattern = branches.first().map(|s| s.as_str()).unwrap_or(pattern);

    // Split by ~ at the top level (respecting parentheses depth AND quoted strings)
    let mut depth = 0;
    let mut in_string = false;
    let mut current = String::new();
    let mut parts = Vec::new();
    let mut prev_char = ' ';

    for ch in pattern.chars() {
        match ch {
            '"' if prev_char != '\\' => {
                in_string = !in_string;
                current.push(ch);
            }
            '(' if !in_string => {
                depth += 1;
                current.push(ch);
            }
            ')' if !in_string => {
                depth -= 1;
                current.push(ch);
            }
            '~' if depth == 0 && !in_string => {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
        prev_char = ch;
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Check modifiers
        let optional = part.ends_with('?') || part.ends_with(")?");
        let repeated = part.ends_with('*')
            || part.ends_with('+')
            || part.ends_with(")*")
            || part.ends_with(")+");

        // Check if this is a group
        if part.starts_with('(')
            && (part.ends_with(')')
                || part.ends_with(")?")
                || part.ends_with(")*")
                || part.ends_with(")+"))
        {
            // For a group, find the rule inside (ignoring literals)
            let inner = if part.ends_with(")?") || part.ends_with(")*") || part.ends_with(")+") {
                &part[1..part.len() - 2]
            } else {
                &part[1..part.len() - 1]
            };

            // Handle alternation inside groups - take first branch
            let inner_branches = split_top_level_alternation(inner);
            let inner = inner_branches.first().map(|s| s.as_str()).unwrap_or(inner);

            // Find the rule reference inside the group (skip literals)
            let inner_parts: Vec<&str> = inner.split('~').map(|s| s.trim()).collect();
            let mut found_rule = None;
            for inner_part in inner_parts {
                let clean = inner_part.trim_end_matches(['?', '*', '+']);
                if !clean.starts_with('"') && !clean.starts_with('\'') && !clean.is_empty() {
                    if !clean
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        found_rule = Some(clean.to_string());
                        break;
                    }
                }
            }

            if let Some(rule_name) = found_rule {
                elements.push(PatternElement {
                    index,
                    kind: PatternKind::Rule(rule_name),
                    name: None,
                    optional,
                    repeated,
                });
            } else {
                // Group with only literals - still counts as a position but no capture
                elements.push(PatternElement {
                    index,
                    kind: PatternKind::Literal("group".to_string()),
                    name: None,
                    optional,
                    repeated,
                });
            }
            index += 1;
        } else {
            // Single element
            let clean = part.trim_end_matches(['?', '*', '+']);

            let kind = if clean.starts_with('"') || clean.starts_with('\'') {
                // String literal
                let literal = clean.trim_matches(|c| c == '"' || c == '\'');
                PatternKind::Literal(literal.to_string())
            } else if clean
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                // Builtin (ASCII_DIGIT, ANY, etc.)
                PatternKind::Builtin(clean.to_string())
            } else if !clean.is_empty() {
                // Rule reference
                PatternKind::Rule(clean.to_string())
            } else {
                index += 1;
                continue;
            };

            elements.push(PatternElement {
                index,
                kind,
                name: None,
                optional,
                repeated,
            });
            index += 1;
        }
    }

    elements
}

/// Sanitize a rule name to be a valid Rust identifier
fn sanitize_rule_name(name: &str) -> String {
    // Remove any alternation or sequence operators, take first valid identifier
    let name = name.split('|').next().unwrap_or(name).trim();
    let name = name.split('~').next().unwrap_or(name).trim();
    // Remove parentheses and modifiers
    let name = name.trim_matches(|c| {
        c == '('
            || c == ')'
            || c == '?'
            || c == '*'
            || c == '+'
            || c == '"'
            || c == '\''
            || c == ' '
    });
    // Replace any remaining invalid chars with underscore
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

/// Generate code to extract children from the parse tree
///
/// IMPORTANT: pest does NOT capture string literals as children.
/// Only named rules become children in the parse tree.
///
/// This function now generates simpler code that collects all children
/// and provides access by rule type, avoiding complex position mapping.
fn generate_child_extraction(pattern: &[PatternElement]) -> String {
    // Collect unique rule names from the pattern
    let mut rule_names: Vec<(String, bool, bool)> = Vec::new(); // (name, optional, repeated)

    for elem in pattern {
        if let PatternKind::Rule(name) = &elem.kind {
            let sanitized = sanitize_rule_name(name);
            if sanitized.is_empty() {
                continue;
            }
            // Avoid duplicates
            if !rule_names.iter().any(|(n, _, _)| n == &sanitized) {
                rule_names.push((sanitized, elem.optional, elem.repeated));
            }
        }
    }

    let mut code = String::new();
    code.push_str("let all_children: Vec<_> = children.collect();\n            ");
    code.push_str("let mut child_iter = all_children.iter();\n            ");

    for (name, optional, repeated) in &rule_names {
        let var_name = format!("child_{}", name);
        if *repeated {
            code.push_str(&format!(
                "let {}: Vec<_> = all_children.iter().filter(|p| p.as_rule() == Rule::{}).cloned().collect();\n            ",
                var_name, name
            ));
        } else if *optional {
            code.push_str(&format!(
                "let {} = all_children.iter().find(|p| p.as_rule() == Rule::{}).cloned();\n            ",
                var_name, name
            ));
        } else {
            code.push_str(&format!(
                "let {} = all_children.iter().find(|p| p.as_rule() == Rule::{}).cloned();\n            ",
                var_name, name
            ));
        }
    }

    code
}

/// Transform capture references ($1, $2, etc.) to proper variable access
///
/// Now uses rule-name-based variables (child_identifier, child_expr, etc.)
/// instead of position-based ones, which is more reliable.
fn transform_captures_to_vars(value: &str, pattern: &[PatternElement]) -> String {
    let mut result = value.to_string();

    // First, replace span($X, $Y) patterns with just `span`
    // since we pre-compute span at the start of each build method
    // Use simple string search to find and replace span(...) calls
    while let Some(start) = result.find("span(") {
        if let Some(end) = result[start..].find(')') {
            let end_pos = start + end + 1;
            result = format!("{}{}{}", &result[..start], "span", &result[end_pos..]);
        } else {
            break;
        }
    }

    // First, flatten groups to get the true element order
    // For pattern like: "const" ~ identifier ~ (":" ~ type_expr)? ~ "=" ~ expr ~ ";"
    // The effective positions are:
    // $1 = "const" (literal)
    // $2 = identifier
    // $3 = type_expr (inside the group)
    // $4 = "=" (literal)
    // $5 = expr
    // $6 = ";" (literal)

    // Build a mapping from position to (rule_name, is_optional, is_repeated)
    let mut position_to_rule: Vec<Option<(String, bool, bool)>> = Vec::new();

    for elem in pattern {
        match &elem.kind {
            PatternKind::Literal(_) => {
                // Literals don't produce children but do occupy a position
                position_to_rule.push(None);
            }
            PatternKind::Rule(name) => {
                let sanitized = sanitize_rule_name(name);
                if !sanitized.is_empty() {
                    position_to_rule.push(Some((sanitized, elem.optional, elem.repeated)));
                } else {
                    position_to_rule.push(None);
                }
            }
            PatternKind::Builtin(_) => {
                // Builtins typically don't produce children
                position_to_rule.push(None);
            }
        }
    }

    // Process from $9 down to $1 to avoid $1 matching $10, etc.
    for i in (1..=9).rev() {
        let pattern_str = format!("${}", i);
        let pattern_with_space = format!("$ {}", i); // TokenStream may insert spaces

        // First, normalize any `$ N` to `$N` for easier processing
        result = result.replace(&pattern_with_space, &pattern_str);

        if !result.contains(&pattern_str) {
            continue;
        }

        // Find what kind of element this is (1-indexed, so i-1)
        if let Some(Some((rule_name, optional, repeated))) = position_to_rule.get(i - 1) {
            let var_name = format!("child_{}", rule_name);

            // Replace intern($N) with getting text from the child as InternedString
            let intern_pattern = format!("intern({})", pattern_str);
            if result.contains(&intern_pattern) {
                let replacement = format!(
                    "InternedString::new_global({}.as_ref().map(|p| p.as_str()).unwrap_or(\"\"))",
                    var_name
                );
                result = result.replace(&intern_pattern, &replacement);
            }

            // Replace $N.collect() with proper iteration and building
            let collect_pattern = format!("{}.collect()", pattern_str);
            if result.contains(&collect_pattern) {
                let replacement = format!(
                    "{}.iter().filter_map(|p| self.build_{}(p.clone()).ok()).collect()",
                    var_name, rule_name
                );
                result = result.replace(&collect_pattern, &replacement);
            }

            // Replace $N.into_iter().map(|...|...) with proper iteration
            // e.g., $2.into_iter().map(|(_, p)| p) -> child_X.iter().filter_map(|p| self.build_X(p.clone()).ok()).map(|(_, p)| p)
            let into_iter_map_pattern = format!("{}.into_iter().map(|", pattern_str);
            if result.contains(&into_iter_map_pattern) {
                // For repeated elements with into_iter, build each element first
                // The result is something like: child_fn_param.iter().filter_map(...).collect()
                let replacement = format!(
                    "{}.iter().filter_map(|p| self.build_{}(p.clone()).ok()).collect::<Vec<_>>().into_iter().map(|",
                    var_name, rule_name
                );
                result = result.replace(&into_iter_map_pattern, &replacement);
            }

            // Replace $N.map(|t| t) with proper optional handling
            let map_pattern = format!("{}.map(|t| t)", pattern_str);
            if result.contains(&map_pattern) {
                let replacement = format!(
                    "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok())",
                    var_name, rule_name
                );
                result = result.replace(&map_pattern, &replacement);
            }

            // Replace $N.unwrap_or_default() pattern
            let unwrap_pattern = format!("{}.unwrap_or_default()", pattern_str);
            if result.contains(&unwrap_pattern) {
                let replacement = format!(
                    "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok()).unwrap_or_default()",
                    var_name, rule_name
                );
                result = result.replace(&unwrap_pattern, &replacement);
            }

            // Handle $N.map(|...|...) - general map patterns with custom closures
            // e.g., $2.map(|s| parse_int(s)) -> build the value and apply the closure
            let general_map_pattern = format!("{}.map(|", pattern_str);
            while result.contains(&general_map_pattern) {
                let start = result.find(&general_map_pattern).unwrap();
                // Find the closing paren of the map call
                let after_map = start + general_map_pattern.len();
                let mut depth = 1;
                let mut end = after_map;
                for (i, ch) in result[after_map..].chars().enumerate() {
                    match ch {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                end = after_map + i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                // Extract the closure body: "|arg| body)"
                let closure_full = &result[after_map..end];
                // Replace $N.map(|x| ...) with build_value.map(|x| ...)
                let replacement = format!(
                    "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok()).map(|{}",
                    var_name, rule_name, closure_full
                );
                result = format!("{}{}{}", &result[..start], replacement, &result[end..]);
            }

            // Replace $N.field patterns (e.g., $7.statements) with building and field access
            let dot_pattern = format!("{}.", pattern_str);
            while let Some(start) = result.find(&dot_pattern) {
                let after_dot = start + dot_pattern.len();
                // Find the field name (alphanumeric chars after the dot)
                let field_end = result[after_dot..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .count();
                if field_end > 0 {
                    let field_name = &result[after_dot..after_dot + field_end];
                    // Skip if it's a method call (map, and_then, into_iter, etc.) - these are handled specially
                    if field_name == "map"
                        || field_name == "and_then"
                        || field_name == "unwrap_or"
                        || field_name == "into_iter"
                        || field_name == "iter"
                        || field_name == "collect"
                        || field_name == "ok"
                        || field_name == "unwrap_or_default"
                        || field_name == "clone"
                    {
                        break;
                    }
                    let full_match_len = dot_pattern.len() + field_end;
                    let replacement = format!(
                        "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok()).map(|v| v.{}).unwrap_or_default()",
                        var_name, rule_name, field_name
                    );
                    result = format!(
                        "{}{}{}",
                        &result[..start],
                        replacement,
                        &result[start + full_match_len..]
                    );
                } else {
                    break;
                }
            }

            // Replace plain $N with the properly typed value
            if *repeated {
                // For repeated elements, build each and collect
                let replacement = format!(
                    "{}.iter().filter_map(|p| self.build_{}(p.clone()).ok()).collect::<Vec<_>>()",
                    var_name, rule_name
                );
                result = result.replace(&pattern_str, &replacement);
            } else if *optional {
                // For optional elements, build if present
                let replacement = format!(
                    "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok())",
                    var_name, rule_name
                );
                result = result.replace(&pattern_str, &replacement);
            } else {
                // For required elements, build and unwrap with default fallback
                let replacement = format!(
                    "{}.as_ref().and_then(|p| self.build_{}(p.clone()).ok()).unwrap_or_default()",
                    var_name, rule_name
                );
                result = result.replace(&pattern_str, &replacement);
            }
        } else if let Some(None) = position_to_rule.get(i - 1) {
            // This position is a literal - it doesn't produce a child, so $N refers to the text
            // Note: pair_str is captured before pair.into_inner() consumes it
            if i == 1 {
                // For $1 referring to a literal (e.g., the matched keyword), use pair_str
                result = result.replace(&pattern_str, "pair_str");
            } else {
                // Other literal positions - use pair_str
                result = result.replace(&pattern_str, "pair_str");
            }
        } else {
            // Position not found in pattern analysis - use a placeholder
            // This happens when the pattern has more elements than we parsed
            // (e.g., complex nested patterns or alternations)
            if result.contains(&format!("{}.unwrap_or_default()", pattern_str)) {
                result = result.replace(
                    &format!("{}.unwrap_or_default()", pattern_str),
                    "Default::default()",
                );
            } else if result.contains(&format!("{}.span", pattern_str)) {
                // $1.span -> span (the local span variable)
                result = result.replace(&format!("{}.span", pattern_str), "span");
            } else if i == 1 {
                // $1 with no matching rule often means "the whole match" (the pair's text)
                // This happens when pattern is all literals, or when $1 refers to the matched string
                result = result.replace(&pattern_str, "pair_str");
            } else {
                // Replace with pair_str as fallback
                result = result.replace(&pattern_str, "pair_str");
            }
        }
    }

    result
}

/// Parse imports string into TokenStream
fn parse_imports(code: &str) -> TokenStream {
    if code.is_empty() {
        return quote! {};
    }

    code.parse().unwrap_or_else(|_| {
        quote! {
            // Failed to parse imports
        }
    })
}

/// Generate context field declarations
fn generate_context_fields(context: &[crate::ContextVar]) -> TokenStream {
    let fields: Vec<TokenStream> = context
        .iter()
        .map(|v| {
            let name = format_ident!("{}", v.name);
            let ty: TokenStream = v.ty.parse().unwrap_or_else(|_| quote! { () });
            quote! { pub #name: #ty }
        })
        .collect();

    quote! { #(#fields,)* }
}

/// Parse type helpers code
fn parse_type_helpers(code: &str) -> TokenStream {
    if code.is_empty() {
        return quote! {};
    }

    code.parse().unwrap_or_else(|_| {
        quote! {
            // Failed to parse type helpers
        }
    })
}

/// Convert string to PascalCase
fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in s.chars() {
        if c == '_' || c == '-' || c == ' ' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c.to_ascii_lowercase());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello"), "Hello");
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("zig"), "Zig");
        assert_eq!(to_pascal_case("my_parser"), "MyParser");
    }

    #[test]
    fn test_transform_captures_to_vars() {
        let pattern = vec![
            PatternElement {
                index: 1,
                kind: PatternKind::Rule("identifier".to_string()),
                name: None,
                optional: false,
                repeated: false,
            },
            PatternElement {
                index: 2,
                kind: PatternKind::Rule("expr".to_string()),
                name: None,
                optional: false,
                repeated: false,
            },
        ];
        // Basic capture reference replacement - now uses rule-name-based variables
        assert!(transform_captures_to_vars("$1", &pattern).contains("child_identifier"));
        assert!(transform_captures_to_vars("$1 + $2", &pattern).contains("child_identifier"));
        assert!(transform_captures_to_vars("$1 + $2", &pattern).contains("child_expr"));
    }

    #[test]
    fn test_analyze_fn_decl_pattern() {
        // This pattern is from fn_decl in zig.zyn
        let pattern = r#""fn" ~ identifier ~
    "(" ~ fn_params? ~ ")" ~ type_expr ~
    block"#;

        let elements = analyze_pattern(pattern);

        // Debug: print what was parsed
        for elem in &elements {
            println!(
                "index={}, kind={:?}, optional={}",
                elem.index, elem.kind, elem.optional
            );
        }

        // There should be 7 elements total
        assert_eq!(elements.len(), 7, "Expected 7 pattern elements");

        // Check each element
        assert!(
            matches!(elements[0].kind, PatternKind::Literal(_)),
            "Element 0 should be 'fn' literal"
        );
        assert!(
            matches!(elements[1].kind, PatternKind::Rule(ref n) if n == "identifier"),
            "Element 1 should be identifier"
        );
        assert!(
            matches!(elements[2].kind, PatternKind::Literal(_)),
            "Element 2 should be '(' literal"
        );
        assert!(
            matches!(elements[3].kind, PatternKind::Rule(ref n) if n == "fn_params"),
            "Element 3 should be fn_params"
        );
        assert!(elements[3].optional, "fn_params should be optional");
        assert!(
            matches!(elements[4].kind, PatternKind::Literal(_)),
            "Element 4 should be ')' literal"
        );
        assert!(
            matches!(elements[5].kind, PatternKind::Rule(ref n) if n == "type_expr"),
            "Element 5 should be type_expr"
        );
        assert!(
            matches!(elements[6].kind, PatternKind::Rule(ref n) if n == "block"),
            "Element 6 should be block"
        );
    }

    #[test]
    fn test_generate_pest_grammar() {
        let grammar = ZynGrammar {
            language: crate::LanguageInfo {
                name: "Test".to_string(),
                ..Default::default()
            },
            rules: vec![
                RuleDef {
                    name: "number".to_string(),
                    modifier: Some(RuleModifier::Atomic),
                    pattern: "ASCII_DIGIT+".to_string(),
                    action: None,
                },
                RuleDef {
                    name: "expr".to_string(),
                    modifier: None,
                    pattern: "number | \"(\" ~ expr ~ \")\"".to_string(),
                    action: None,
                },
            ],
            ..Default::default()
        };

        let pest = generate_pest_grammar(&grammar).unwrap();
        assert!(pest.contains("number = @{ ASCII_DIGIT+ }"));
        assert!(pest.contains("expr = { number | \"(\" ~ expr ~ \")\" }"));
        assert!(pest.contains("WHITESPACE"));
    }
}
