//! TypedAST Construction Helpers
//!
//! Helper functions for building TypedAST nodes from parsed values.
//! These functions bridge the gap between parser output and TypedAST construction.

use zyntax_typed_ast::{
    type_registry::{CallingConvention, Mutability, PrimitiveType, Type, TypeRegistry, Visibility},
    typed_ast::{
        ParameterKind, TypedBinary, TypedCall, TypedFieldAccess, TypedFieldInit, TypedFor, TypedIf,
        TypedLambda, TypedLambdaBody, TypedLambdaParam, TypedLet, TypedLiteral, TypedMethodCall,
        TypedParameter, TypedPattern, TypedStructLiteral, TypedTypeParam, TypedUnary, TypedWhile,
    },
    BinaryOp, InternedString, Span, TypedASTBuilder, TypedBlock, TypedClass, TypedDeclaration,
    TypedEnum, TypedExpression, TypedField, TypedFunction, TypedNode, TypedProgram, TypedStatement,
    TypedVariable, UnaryOp,
};

use crate::runtime2::state::{NodeHandle, ParsedValue};

/// Helper context for TypedAST construction
pub struct AstContext<'a> {
    pub builder: &'a mut TypedASTBuilder,
    pub type_registry: &'a mut TypeRegistry,
}

impl<'a> AstContext<'a> {
    pub fn new(builder: &'a mut TypedASTBuilder, type_registry: &'a mut TypeRegistry) -> Self {
        AstContext {
            builder,
            type_registry,
        }
    }

    /// Intern a string
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.builder.intern(s)
    }

    /// Create a span
    pub fn span(&self, start: usize, end: usize) -> Span {
        Span::new(start, end)
    }

    /// Wrap a node with type and span
    pub fn node<T>(&self, value: T, ty: Type, span: Span) -> TypedNode<T> {
        TypedNode {
            node: value,
            ty,
            span,
        }
    }

    /// Create a node with unknown type
    pub fn node_unknown<T>(&self, value: T, span: Span) -> TypedNode<T> {
        TypedNode {
            node: value,
            ty: Type::Unknown,
            span,
        }
    }
}

// =============================================================================
// Expression Builders
// =============================================================================

/// Create an integer literal expression
pub fn int_literal(value: i64) -> TypedExpression {
    TypedExpression::Literal(TypedLiteral::Integer(value as i128))
}

/// Create a float literal expression
pub fn float_literal(value: f64) -> TypedExpression {
    TypedExpression::Literal(TypedLiteral::Float(value))
}

/// Create a string literal expression (requires InternedString)
pub fn string_literal(value: InternedString) -> TypedExpression {
    TypedExpression::Literal(TypedLiteral::String(value))
}

/// Create a boolean literal expression
pub fn bool_literal(value: bool) -> TypedExpression {
    TypedExpression::Literal(TypedLiteral::Bool(value))
}

/// Create a variable reference expression
pub fn variable(name: InternedString) -> TypedExpression {
    TypedExpression::Variable(name)
}

/// Create a binary expression
pub fn binary(
    left: TypedNode<TypedExpression>,
    op: BinaryOp,
    right: TypedNode<TypedExpression>,
) -> TypedExpression {
    TypedExpression::Binary(TypedBinary {
        left: Box::new(left),
        op,
        right: Box::new(right),
    })
}

/// Create a unary expression
pub fn unary(op: UnaryOp, operand: TypedNode<TypedExpression>) -> TypedExpression {
    TypedExpression::Unary(TypedUnary {
        op,
        operand: Box::new(operand),
    })
}

/// Create a function call expression
pub fn call(
    callee: TypedNode<TypedExpression>,
    args: Vec<TypedNode<TypedExpression>>,
) -> TypedExpression {
    TypedExpression::Call(TypedCall {
        callee: Box::new(callee),
        positional_args: args,
        named_args: Vec::new(),
        type_args: Vec::new(),
    })
}

/// Create a method call expression
pub fn method_call(
    receiver: TypedNode<TypedExpression>,
    method: InternedString,
    args: Vec<TypedNode<TypedExpression>>,
) -> TypedExpression {
    TypedExpression::MethodCall(TypedMethodCall {
        receiver: Box::new(receiver),
        method,
        positional_args: args,
        named_args: Vec::new(),
        type_args: Vec::new(),
    })
}

/// Create a field access expression
pub fn field_access(object: TypedNode<TypedExpression>, field: InternedString) -> TypedExpression {
    TypedExpression::Field(TypedFieldAccess {
        object: Box::new(object),
        field,
    })
}

/// Create a struct literal expression
pub fn struct_literal(
    name: InternedString,
    fields: Vec<(InternedString, TypedNode<TypedExpression>)>,
) -> TypedExpression {
    TypedExpression::Struct(TypedStructLiteral {
        name,
        fields: fields
            .into_iter()
            .map(|(name, value)| TypedFieldInit {
                name,
                value: Box::new(value),
            })
            .collect(),
    })
}

/// Create an array literal expression
pub fn array_literal(elements: Vec<TypedNode<TypedExpression>>) -> TypedExpression {
    TypedExpression::Array(elements)
}

/// Create a lambda expression
pub fn lambda(params: Vec<TypedLambdaParam>, body: TypedNode<TypedExpression>) -> TypedExpression {
    TypedExpression::Lambda(TypedLambda {
        params,
        body: TypedLambdaBody::Expression(Box::new(body)),
        captures: Vec::new(),
    })
}

/// Create a lambda with block body
pub fn lambda_block(params: Vec<TypedLambdaParam>, body: TypedBlock) -> TypedExpression {
    TypedExpression::Lambda(TypedLambda {
        params,
        body: TypedLambdaBody::Block(body),
        captures: Vec::new(),
    })
}

// =============================================================================
// Statement Builders
// =============================================================================

/// Create a let statement
pub fn let_stmt(
    name: InternedString,
    ty: Type,
    value: Option<TypedNode<TypedExpression>>,
    mutable: bool,
    span: Span,
) -> TypedStatement {
    TypedStatement::Let(TypedLet {
        name,
        ty,
        mutability: if mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        },
        initializer: value.map(Box::new),
        span,
    })
}

/// Create a return statement
pub fn return_stmt(value: Option<TypedNode<TypedExpression>>) -> TypedStatement {
    TypedStatement::Return(value.map(Box::new))
}

/// Create an if statement
pub fn if_stmt(
    condition: TypedNode<TypedExpression>,
    then_block: TypedBlock,
    else_block: Option<TypedBlock>,
    span: Span,
) -> TypedStatement {
    TypedStatement::If(TypedIf {
        condition: Box::new(condition),
        then_block,
        else_block,
        span,
    })
}

/// Create a while statement
pub fn while_stmt(
    condition: TypedNode<TypedExpression>,
    body: TypedBlock,
    span: Span,
) -> TypedStatement {
    TypedStatement::While(TypedWhile {
        condition: Box::new(condition),
        body,
        span,
    })
}

/// Create a for statement
pub fn for_stmt(
    pattern: TypedNode<TypedPattern>,
    iterator: TypedNode<TypedExpression>,
    body: TypedBlock,
) -> TypedStatement {
    TypedStatement::For(TypedFor {
        pattern: Box::new(pattern),
        iterator: Box::new(iterator),
        body,
    })
}

/// Create an expression statement
pub fn expr_stmt(expr: TypedNode<TypedExpression>) -> TypedStatement {
    TypedStatement::Expression(Box::new(expr))
}

/// Create a block statement
pub fn block_stmt(statements: Vec<TypedNode<TypedStatement>>, span: Span) -> TypedStatement {
    TypedStatement::Block(TypedBlock { statements, span })
}

/// Create a block
pub fn block(statements: Vec<TypedNode<TypedStatement>>, span: Span) -> TypedBlock {
    TypedBlock { statements, span }
}

// =============================================================================
// Declaration Builders
// =============================================================================

/// Create a function declaration
pub fn function_decl(
    name: InternedString,
    params: Vec<TypedParameter>,
    return_type: Type,
    body: Option<TypedBlock>,
    type_params: Vec<TypedTypeParam>,
) -> TypedDeclaration {
    TypedDeclaration::Function(TypedFunction {
        name,
        annotations: vec![],
        effects: vec![],
        params,
        return_type,
        body,
        visibility: Visibility::Public,
        type_params,
        is_async: false,
        is_pure: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
    })
}

/// Create a variable declaration
pub fn variable_decl(
    name: InternedString,
    ty: Type,
    value: Option<TypedNode<TypedExpression>>,
    mutable: bool,
) -> TypedDeclaration {
    TypedDeclaration::Variable(TypedVariable {
        name,
        ty,
        initializer: value.map(Box::new),
        mutability: if mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        },
        visibility: Visibility::Private,
    })
}

/// Create a class/struct declaration
pub fn class_decl(
    name: InternedString,
    fields: Vec<TypedField>,
    type_params: Vec<TypedTypeParam>,
    span: Span,
) -> TypedDeclaration {
    TypedDeclaration::Class(TypedClass {
        name,
        fields,
        methods: Vec::new(),
        type_params,
        visibility: Visibility::Public,
        extends: None,
        implements: Vec::new(),
        constructors: Vec::new(),
        is_abstract: false,
        is_final: false,
        span,
    })
}

// =============================================================================
// Operator Helpers
// =============================================================================

/// Parse a binary operator from string
pub fn parse_binary_op(op: &str) -> BinaryOp {
    match op {
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        "%" => BinaryOp::Rem,
        "==" => BinaryOp::Eq,
        "!=" => BinaryOp::Ne,
        "<" => BinaryOp::Lt,
        ">" => BinaryOp::Gt,
        "<=" => BinaryOp::Le,
        ">=" => BinaryOp::Ge,
        "&&" | "and" => BinaryOp::And,
        "||" | "or" => BinaryOp::Or,
        "&" => BinaryOp::BitAnd,
        "|" => BinaryOp::BitOr,
        "^" => BinaryOp::BitXor,
        "<<" => BinaryOp::Shl,
        ">>" => BinaryOp::Shr,
        "=" => BinaryOp::Assign,
        "orelse" => BinaryOp::Orelse,
        "catch" => BinaryOp::Catch,
        _ => BinaryOp::Add, // fallback
    }
}

/// Parse a unary operator from string
pub fn parse_unary_op(op: &str) -> UnaryOp {
    match op {
        "-" => UnaryOp::Minus,
        "+" => UnaryOp::Plus,
        "!" => UnaryOp::Not,
        "~" => UnaryOp::BitNot,
        _ => UnaryOp::Minus, // fallback
    }
}

// =============================================================================
// Type Helpers
// =============================================================================

/// Parse a type from string
pub fn parse_type(name: &str, ctx: &mut AstContext) -> Type {
    match name {
        "i8" => Type::Primitive(PrimitiveType::I8),
        "i16" => Type::Primitive(PrimitiveType::I16),
        "i32" => Type::Primitive(PrimitiveType::I32),
        "i64" => Type::Primitive(PrimitiveType::I64),
        "u8" => Type::Primitive(PrimitiveType::U8),
        "u16" => Type::Primitive(PrimitiveType::U16),
        "u32" => Type::Primitive(PrimitiveType::U32),
        "u64" => Type::Primitive(PrimitiveType::U64),
        "f32" => Type::Primitive(PrimitiveType::F32),
        "f64" => Type::Primitive(PrimitiveType::F64),
        "bool" => Type::Primitive(PrimitiveType::Bool),
        "char" => Type::Primitive(PrimitiveType::Char),
        "String" | "str" => Type::Primitive(PrimitiveType::String),
        "()" | "unit" => Type::Primitive(PrimitiveType::Unit),
        "never" | "!" => Type::Never,
        _ => {
            let interned = ctx.intern(name);
            Type::Unresolved(interned)
        }
    }
}

// =============================================================================
// Fold Helpers for Binary Expressions
// =============================================================================

/// Fold a list of (operator, operand) pairs into left-associative binary expressions
pub fn fold_binary_left(
    first: TypedNode<TypedExpression>,
    rest: Vec<(BinaryOp, TypedNode<TypedExpression>)>,
) -> TypedNode<TypedExpression> {
    let mut result = first;
    for (op, right) in rest {
        let span = Span::new(result.span.start, right.span.end);
        let expr = binary(result, op, right);
        result = TypedNode {
            node: expr,
            ty: Type::Unknown,
            span,
        };
    }
    result
}

/// Fold a list of (operand, operator) pairs into right-associative binary expressions
pub fn fold_binary_right(
    pairs: Vec<(TypedNode<TypedExpression>, BinaryOp)>,
    last: TypedNode<TypedExpression>,
) -> TypedNode<TypedExpression> {
    let mut result = last;
    for (left, op) in pairs.into_iter().rev() {
        let span = Span::new(left.span.start, result.span.end);
        let expr = binary(left, op, result);
        result = TypedNode {
            node: expr,
            ty: Type::Unknown,
            span,
        };
    }
    result
}

// =============================================================================
// ParsedValue Conversion Helpers
// =============================================================================

/// Extract text from a ParsedValue
pub fn value_to_text(value: &ParsedValue) -> Option<String> {
    match value {
        ParsedValue::Text(s) => Some(s.clone()),
        ParsedValue::Interned(s) => s.resolve_global().map(|s| s.to_string()),
        _ => None,
    }
}

/// Extract integer from a ParsedValue
pub fn value_to_int(value: &ParsedValue) -> Option<i64> {
    match value {
        ParsedValue::Int(n) => Some(*n),
        ParsedValue::Text(s) => s.parse().ok(),
        _ => None,
    }
}

/// Extract boolean from a ParsedValue
pub fn value_to_bool(value: &ParsedValue) -> Option<bool> {
    match value {
        ParsedValue::Bool(b) => Some(*b),
        ParsedValue::Text(s) => match s.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Extract list from a ParsedValue
pub fn value_to_list(value: &ParsedValue) -> Option<&Vec<ParsedValue>> {
    match value {
        ParsedValue::List(items) => Some(items),
        _ => None,
    }
}

/// Check if ParsedValue is Some (for optionals)
pub fn value_is_some(value: &ParsedValue) -> bool {
    match value {
        ParsedValue::Optional(opt) => opt.is_some(),
        ParsedValue::None => false,
        _ => true,
    }
}

/// Unwrap optional ParsedValue
pub fn value_unwrap_optional(value: &ParsedValue) -> Option<&ParsedValue> {
    match value {
        ParsedValue::Optional(Some(inner)) => Some(inner.as_ref()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_binary_op() {
        assert_eq!(parse_binary_op("+"), BinaryOp::Add);
        assert_eq!(parse_binary_op("=="), BinaryOp::Eq);
        assert_eq!(parse_binary_op("&&"), BinaryOp::And);
        assert_eq!(parse_binary_op("%"), BinaryOp::Rem);
    }

    #[test]
    fn test_parse_unary_op() {
        assert_eq!(parse_unary_op("-"), UnaryOp::Minus);
        assert_eq!(parse_unary_op("!"), UnaryOp::Not);
        assert_eq!(parse_unary_op("+"), UnaryOp::Plus);
    }

    #[test]
    fn test_value_conversions() {
        assert_eq!(
            value_to_text(&ParsedValue::Text("hello".into())),
            Some("hello".into())
        );
        assert_eq!(value_to_int(&ParsedValue::Int(42)), Some(42));
        assert_eq!(value_to_bool(&ParsedValue::Bool(true)), Some(true));
    }

    #[test]
    fn test_value_is_some() {
        assert!(value_is_some(&ParsedValue::Text("x".into())));
        assert!(!value_is_some(&ParsedValue::None));
        assert!(value_is_some(&ParsedValue::Optional(Some(Box::new(
            ParsedValue::Int(1)
        )))));
        assert!(!value_is_some(&ParsedValue::Optional(None)));
    }
}
