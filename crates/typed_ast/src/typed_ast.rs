//! # Typed Abstract Syntax Tree (TypedAST)
//!
//! The target representation that other language implementations will lower to.
//! Includes full type information and source locations for debugging and diagnostics.
//!
//! ## Design Principles
//! - Every node carries source location (Span) information
//! - Variable declarations include mutability information
//! - Built incrementally to avoid compilation errors
//! - Supports languages like Rust, Java, C#, TypeScript, and Haxe

use crate::arena::InternedString;
use crate::source::{SourceFile, Span};
use crate::type_registry::{CallingConvention, Mutability, Type, Visibility};
use serde::{Deserialize, Serialize};

/// Every typed node wraps its content with type and span information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedNode<T> {
    pub node: T,
    pub ty: Type,
    pub span: Span,
}

impl<T> TypedNode<T> {
    pub fn new(node: T, ty: Type, span: Span) -> Self {
        Self { node, ty, span }
    }
}

impl<T: Default> Default for TypedNode<T> {
    fn default() -> Self {
        Self {
            node: T::default(),
            ty: Type::Never,
            span: Span::default(),
        }
    }
}

/// Typed program - the root of the AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedProgram {
    pub declarations: Vec<TypedNode<TypedDeclaration>>,
    #[serde(default)]
    pub span: Span,
    /// Source files used in this program (for diagnostics)
    #[serde(default)]
    pub source_files: Vec<SourceFile>,
    /// Type registry for custom struct/enum types defined in the program
    #[serde(skip, default = "crate::TypeRegistry::new")]
    pub type_registry: crate::TypeRegistry,
}

impl Default for TypedProgram {
    fn default() -> Self {
        Self {
            declarations: Vec::new(),
            span: Span::default(),
            source_files: Vec::new(),
            type_registry: crate::TypeRegistry::new(),
        }
    }
}

/// Top-level declarations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedDeclaration {
    Function(TypedFunction),
    Variable(TypedVariable),
    Class(TypedClass),
    Interface(TypedInterface),
    Impl(TypedTraitImpl),
    Enum(TypedEnum),
    TypeAlias(TypedTypeAlias),
    Module(TypedModule),
    Import(TypedImport),
    /// External declaration - type defined outside the compilation unit
    /// whose methods are mapped to runtime symbols
    Extern(TypedExtern),
    /// Algebraic effect declaration: effect Probabilistic { def sample<T>(...): T }
    Effect(TypedEffect),
    /// Effect handler declaration: handler MCMC for Probabilistic { ... }
    EffectHandler(TypedEffectHandler),
}

/// External type declaration - represents types defined outside the compilation unit
///
/// This is used for language-specific extern types like:
/// - Haxe's String, Array, Map classes
/// - C's opaque struct types
/// - FFI type declarations
///
/// The runtime_prefix is used to map methods to runtime symbols:
/// e.g., runtime_prefix = "$haxe$String" means method "length" -> "$haxe$String$length"
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedExtern {
    /// External class with methods mapped to runtime symbols
    Class(TypedExternClass),
    /// External struct (opaque type)
    Struct(TypedExternStruct),
    /// External enum with variants mapped to runtime symbols
    Enum(TypedExternEnum),
    /// External type alias (like C typedef)
    TypeDef(TypedExternTypeDef),
}

/// External class declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternClass {
    /// Class name (e.g., "String", "Array")
    pub name: InternedString,
    /// Runtime symbol prefix for method resolution
    /// e.g., "$haxe$String" -> method "length" becomes "$haxe$String$length"
    pub runtime_prefix: InternedString,
    /// Type parameters for generic extern classes (e.g., Array<T>)
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
    /// Method signatures (the implementations are in the runtime)
    #[serde(default)]
    pub methods: Vec<TypedExternMethod>,
    /// Property signatures
    #[serde(default)]
    pub properties: Vec<TypedExternProperty>,
}

/// External struct (opaque type) declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternStruct {
    pub name: InternedString,
    pub runtime_prefix: InternedString,
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
}

/// External enum declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternEnum {
    pub name: InternedString,
    pub runtime_prefix: InternedString,
    #[serde(default)]
    pub variants: Vec<TypedExternEnumVariant>,
}

/// External enum variant
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternEnumVariant {
    pub name: InternedString,
    /// Runtime symbol for this variant constructor
    pub symbol: InternedString,
    #[serde(default)]
    pub fields: Vec<Type>,
}

/// External typedef
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternTypeDef {
    pub name: InternedString,
    pub target_type: Type,
}

/// External method signature
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternMethod {
    /// Method name in source language
    pub name: InternedString,
    /// Runtime symbol to call (if None, derived from class runtime_prefix + method name)
    pub symbol: Option<InternedString>,
    #[serde(default)]
    pub params: Vec<TypedParameter>,
    #[serde(default)]
    pub return_type: Type,
    /// Whether this is a static method
    #[serde(default)]
    pub is_static: bool,
    /// Whether this method mutates self
    #[serde(default)]
    pub mutates_self: bool,
}

/// External property signature
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedExternProperty {
    /// Property name
    pub name: InternedString,
    /// Getter symbol (if None, derived from runtime_prefix)
    pub getter_symbol: Option<InternedString>,
    /// Setter symbol (None means read-only)
    pub setter_symbol: Option<InternedString>,
    #[serde(default)]
    pub ty: Type,
}

/// Annotation/decorator on items: @deprecated("message"), @inline, @derive(Debug, Clone)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedAnnotation {
    /// Annotation name (e.g., "deprecated", "inline", "derive")
    pub name: InternedString,
    /// Positional arguments (e.g., for @deprecated("message"))
    #[serde(default)]
    pub args: Vec<TypedAnnotationArg>,
    /// Source span
    #[serde(default)]
    pub span: Span,
}

/// Annotation argument - can be positional or named
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedAnnotationArg {
    /// Positional argument: @deprecated("message")
    Positional(TypedAnnotationValue),
    /// Named argument: @validate(min=1, max=100)
    Named {
        name: InternedString,
        value: TypedAnnotationValue,
    },
}

/// Values that can appear in annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedAnnotationValue {
    /// String literal: "message"
    String(InternedString),
    /// Integer literal: 100
    Integer(i64),
    /// Float literal: 0.5
    Float(f64),
    /// Boolean literal: true
    Bool(bool),
    /// Identifier (e.g., Debug, Clone in @derive(Debug, Clone))
    Identifier(InternedString),
    /// Nested annotation list (for @derive(Debug, Clone))
    List(Vec<TypedAnnotationValue>),
}

// ============================================================================
// Algebraic Effects
// ============================================================================

/// Algebraic effect declaration
///
/// ```zyn
/// effect Probabilistic {
///     def sample<T>(dist: Distribution<T>): T
///     def observe<T>(dist: Distribution<T>, value: T)
///     def factor(score: float)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedEffect {
    /// Effect name (e.g., "Probabilistic", "State", "Async")
    pub name: InternedString,
    /// Type parameters (e.g., State<S> has parameter S)
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
    /// Effect operations (sample, observe, factor, etc.)
    #[serde(default)]
    pub operations: Vec<TypedEffectOp>,
    /// Source span
    #[serde(default)]
    pub span: Span,
}

/// Effect operation declaration
///
/// ```zyn
/// def sample<T>(dist: Distribution<T>): T
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedEffectOp {
    /// Operation name (e.g., "sample", "observe", "get", "put")
    pub name: InternedString,
    /// Type parameters for this operation
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
    /// Parameters
    #[serde(default)]
    pub params: Vec<TypedParameter>,
    /// Return type
    #[serde(default)]
    pub return_type: Type,
    /// Source span
    #[serde(default)]
    pub span: Span,
}

/// Effect handler declaration
///
/// ```zyn
/// handler MCMC for Probabilistic {
///     def sample<T>(self, dist: Distribution<T>, resume: Resume<T>): T {
///         // MCMC implementation
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedEffectHandler {
    /// Handler name (e.g., "MCMC", "VariationalInference")
    pub name: InternedString,
    /// Effect being handled (e.g., "Probabilistic")
    pub effect_name: InternedString,
    /// Type parameters
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
    /// Handler fields/state
    #[serde(default)]
    pub fields: Vec<TypedField>,
    /// Handler operation implementations
    #[serde(default)]
    pub handlers: Vec<TypedEffectHandlerImpl>,
    /// Source span
    #[serde(default)]
    pub span: Span,
}

/// Effect handler operation implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedEffectHandlerImpl {
    /// Operation name being handled (e.g., "sample")
    pub op_name: InternedString,
    /// Type parameters
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>,
    /// Parameters (including self and resume)
    #[serde(default)]
    pub params: Vec<TypedParameter>,
    /// Return type
    #[serde(default)]
    pub return_type: Type,
    /// Implementation body
    pub body: Option<TypedBlock>,
    /// Source span
    #[serde(default)]
    pub span: Span,
}

/// Function declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedFunction {
    #[serde(default)]
    pub name: InternedString,
    #[serde(default)]
    pub annotations: Vec<TypedAnnotation>, // @deprecated, @inline, etc.
    #[serde(default)]
    pub effects: Vec<InternedString>, // Effect names from @effect(Probabilistic, IO)
    #[serde(default)]
    pub type_params: Vec<TypedTypeParam>, // Generic type parameters
    #[serde(default)]
    pub params: Vec<TypedParameter>,
    #[serde(default)]
    pub return_type: Type,
    pub body: Option<TypedBlock>, // None for extern functions
    #[serde(default)]
    pub visibility: Visibility,
    #[serde(default)]
    pub is_async: bool,
    #[serde(default)]
    pub is_pure: bool, // True for @pure functions (no effects)
    #[serde(default)]
    pub is_external: bool, // True for extern/foreign functions
    #[serde(default)]
    pub calling_convention: CallingConvention, // Calling convention (C, Rust, System, etc.)
    pub link_name: Option<InternedString>, // Override symbol name for linking
}

/// Function parameter with mutability and advanced features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedParameter {
    #[serde(default)]
    pub name: InternedString,
    #[serde(default)]
    pub ty: Type,
    #[serde(default)]
    pub mutability: Mutability,
    #[serde(default)]
    pub kind: ParameterKind,
    pub default_value: Option<Box<TypedNode<TypedExpression>>>,
    #[serde(default)]
    pub attributes: Vec<ParameterAttribute>,
    #[serde(default)]
    pub span: Span,
}

/// Parameter kinds for different argument passing conventions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ParameterKind {
    /// Regular parameter: func(x: int)
    #[default]
    Regular,
    /// Out parameter (C#): func(out int x)
    Out,
    /// Reference parameter (C#): func(ref int x)
    Ref,
    /// In-out parameter (Swift): func(inout x: Int)
    InOut,
    /// Rest/variadic parameter: func(...args) or func(args: ...int)
    Rest,
    /// Optional parameter with default: func(x: int = 5)
    Optional,
    /// Keyword-only parameter (Python): func(*, x: int)
    KeywordOnly,
    /// Positional-only parameter (Python): func(x: int, /)
    PositionalOnly,
}

/// Parameter attributes for validation and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterAttribute {
    pub name: InternedString,
    pub args: Vec<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// Variable declaration with mutability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedVariable {
    pub name: InternedString,
    pub ty: Type,
    pub mutability: Mutability,
    pub initializer: Option<Box<TypedNode<TypedExpression>>>,
    pub visibility: Visibility,
}

/// Typed statements - all carry span information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedStatement {
    Continue,
    Expression(Box<TypedNode<TypedExpression>>),
    Let(TypedLet),
    /// Let with pattern destructuring: let (x, y) = expr
    LetPattern(TypedLetPattern),
    Return(Option<Box<TypedNode<TypedExpression>>>),
    If(TypedIf),
    While(TypedWhile),
    Block(TypedBlock),
    For(TypedFor),
    ForCStyle(TypedForCStyle),
    Loop(TypedLoop),
    Match(TypedMatch),
    Try(TypedTry),
    Throw(Box<TypedNode<TypedExpression>>),
    Break(Option<Box<TypedNode<TypedExpression>>>),
    Coroutine(TypedCoroutine),
    Defer(TypedDefer),
    Select(TypedSelect),
}

impl Default for TypedStatement {
    fn default() -> Self {
        TypedStatement::Block(TypedBlock::default())
    }
}

/// Let statement with mutability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedLet {
    pub name: InternedString,
    pub ty: Type,
    pub mutability: Mutability,
    pub initializer: Option<Box<TypedNode<TypedExpression>>>,
    pub span: Span,
}

/// Let with pattern destructuring: let (x, y) = expr or let Point { x, y } = point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedLetPattern {
    pub pattern: Box<TypedNode<TypedPattern>>,
    pub initializer: Box<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// If statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedIf {
    pub condition: Box<TypedNode<TypedExpression>>,
    pub then_block: TypedBlock,
    pub else_block: Option<TypedBlock>,
    pub span: Span,
}

/// While loop
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedWhile {
    pub condition: Box<TypedNode<TypedExpression>>,
    pub body: TypedBlock,
    pub span: Span,
}

/// Block of statements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TypedBlock {
    pub statements: Vec<TypedNode<TypedStatement>>,
    #[serde(default)]
    pub span: Span,
}

/// Typed expressions - all carry type and span information via TypedNode
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedExpression {
    Literal(TypedLiteral),
    Variable(InternedString),
    Binary(TypedBinary),
    Unary(TypedUnary),
    Call(TypedCall),
    Field(TypedFieldAccess),
    Index(TypedIndex),
    Array(Vec<TypedNode<TypedExpression>>),
    Tuple(Vec<TypedNode<TypedExpression>>),
    Struct(TypedStructLiteral),
    Lambda(TypedLambda),
    Match(TypedMatchExpr),
    If(TypedIfExpr),
    Cast(TypedCast),
    Await(Box<TypedNode<TypedExpression>>),
    Try(Box<TypedNode<TypedExpression>>),
    Reference(TypedReference),
    Dereference(Box<TypedNode<TypedExpression>>),
    Range(TypedRange),
    MethodCall(TypedMethodCall),
    Block(TypedBlock),
    /// List comprehension: [expr for var in iter if condition]
    ListComprehension(TypedListComprehension),
    /// Slice expression: arr[start:end:step]
    Slice(TypedSlice),
    /// Import modifier expression: import loader("path") as Type
    ImportModifier(TypedImportModifier),
    /// Path expression: Type::method or module::function
    Path(TypedPath),
}

impl Default for TypedExpression {
    fn default() -> Self {
        TypedExpression::Block(TypedBlock::default())
    }
}

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum TypedLiteral {
    Integer(i128),
    Float(f64),
    Bool(bool),
    String(InternedString),
    Char(char),
    #[default]
    Unit,
    /// Null literal for optional types (null in ?T)
    Null,
    /// Undefined literal for uninitialized memory
    Undefined,
}

/// Binary operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedBinary {
    pub op: BinaryOp,
    pub left: Box<TypedNode<TypedExpression>>,
    pub right: Box<TypedNode<TypedExpression>>,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    // Assignment (with mutability check)
    Assign,
    // Zig-specific error handling
    Orelse, // `a orelse b` - unwrap optional or use default
    Catch,  // `a catch b` - unwrap error union or use default
}

/// Unary operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedUnary {
    pub op: UnaryOp,
    pub operand: Box<TypedNode<TypedExpression>>,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
    BitNot,
}

/// Function call with support for both positional and named arguments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedCall {
    pub callee: Box<TypedNode<TypedExpression>>,
    pub positional_args: Vec<TypedNode<TypedExpression>>,
    pub named_args: Vec<TypedNamedArg>,
    pub type_args: Vec<Type>, // Generic type arguments
}

/// Named argument in function call
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedNamedArg {
    pub name: InternedString,
    pub value: Box<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// Field access
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedFieldAccess {
    pub object: Box<TypedNode<TypedExpression>>,
    pub field: InternedString,
}

/// Array/slice indexing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedIndex {
    pub object: Box<TypedNode<TypedExpression>>,
    pub index: Box<TypedNode<TypedExpression>>,
}

// Note: All types are already public at the module level, no need for re-exports

/// Helper to create a typed node
pub fn typed_node<T>(node: T, ty: Type, span: Span) -> TypedNode<T> {
    TypedNode::new(node, ty, span)
}

// ====== PARAMETER AND CALL CONSTRUCTION HELPERS ======

impl TypedParameter {
    /// Create a regular parameter
    pub fn regular(name: InternedString, ty: Type, mutability: Mutability, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span,
        }
    }

    /// Create an optional parameter with default value
    pub fn optional(
        name: InternedString,
        ty: Type,
        mutability: Mutability,
        default: TypedNode<TypedExpression>,
        span: Span,
    ) -> Self {
        Self {
            name,
            ty,
            mutability,
            kind: ParameterKind::Optional,
            default_value: Some(Box::new(default)),
            attributes: vec![],
            span,
        }
    }

    /// Create a rest/variadic parameter
    pub fn rest(name: InternedString, ty: Type, mutability: Mutability, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability,
            kind: ParameterKind::Rest,
            default_value: None,
            attributes: vec![],
            span,
        }
    }

    /// Create an out parameter (C#-style)
    pub fn out(name: InternedString, ty: Type, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability: Mutability::Mutable, // Out parameters are inherently mutable
            kind: ParameterKind::Out,
            default_value: None,
            attributes: vec![],
            span,
        }
    }

    /// Create a ref parameter (C#-style)
    pub fn ref_param(name: InternedString, ty: Type, mutability: Mutability, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability,
            kind: ParameterKind::Ref,
            default_value: None,
            attributes: vec![],
            span,
        }
    }

    /// Create an inout parameter (Swift-style)
    pub fn inout(name: InternedString, ty: Type, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability: Mutability::Mutable, // InOut parameters are inherently mutable
            kind: ParameterKind::InOut,
            default_value: None,
            attributes: vec![],
            span,
        }
    }
}

impl TypedMethodParam {
    /// Create a regular method parameter
    pub fn regular(name: InternedString, ty: Type, mutability: Mutability, span: Span) -> Self {
        Self {
            name,
            ty,
            mutability,
            is_self: false,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span,
        }
    }

    /// Create a self parameter (requires arena for string interning)
    pub fn self_param(
        arena: &mut crate::arena::AstArena,
        ty: Type,
        mutability: Mutability,
        span: Span,
    ) -> Self {
        Self {
            name: arena.intern_string("self"), // Special self name
            ty,
            mutability,
            is_self: true,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span,
        }
    }
}

impl TypedCall {
    /// Create a call with only positional arguments
    pub fn positional(
        callee: TypedNode<TypedExpression>,
        args: Vec<TypedNode<TypedExpression>>,
    ) -> Self {
        Self {
            callee: Box::new(callee),
            positional_args: args,
            named_args: vec![],
            type_args: vec![],
        }
    }

    /// Create a call with both positional and named arguments
    pub fn mixed(
        callee: TypedNode<TypedExpression>,
        positional: Vec<TypedNode<TypedExpression>>,
        named: Vec<TypedNamedArg>,
        type_args: Vec<Type>,
    ) -> Self {
        Self {
            callee: Box::new(callee),
            positional_args: positional,
            named_args: named,
            type_args,
        }
    }

    /// Create a call with only named arguments (Python/Swift style)
    pub fn named_only(callee: TypedNode<TypedExpression>, args: Vec<TypedNamedArg>) -> Self {
        Self {
            callee: Box::new(callee),
            positional_args: vec![],
            named_args: args,
            type_args: vec![],
        }
    }
}

impl TypedNamedArg {
    /// Create a named argument
    pub fn new(name: InternedString, value: TypedNode<TypedExpression>, span: Span) -> Self {
        Self {
            name,
            value: Box::new(value),
            span,
        }
    }
}

impl TypedMethodCall {
    /// Create a method call with positional arguments
    pub fn positional(
        receiver: TypedNode<TypedExpression>,
        method: InternedString,
        args: Vec<TypedNode<TypedExpression>>,
    ) -> Self {
        Self {
            receiver: Box::new(receiver),
            method,
            type_args: vec![],
            positional_args: args,
            named_args: vec![],
        }
    }

    /// Create a method call with named arguments
    pub fn named(
        receiver: TypedNode<TypedExpression>,
        method: InternedString,
        named: Vec<TypedNamedArg>,
    ) -> Self {
        Self {
            receiver: Box::new(receiver),
            method,
            type_args: vec![],
            positional_args: vec![],
            named_args: named,
        }
    }
}

// ====== EXTENDED EXPRESSION AND STATEMENT TYPES ======

/// Struct literal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedStructLiteral {
    pub name: InternedString,
    pub fields: Vec<TypedFieldInit>,
}

/// Field initialization in struct literal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedFieldInit {
    pub name: InternedString,
    pub value: Box<TypedNode<TypedExpression>>,
}

/// Lambda/Closure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedLambda {
    pub params: Vec<TypedLambdaParam>,
    pub body: TypedLambdaBody,
    pub captures: Vec<TypedCapture>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedLambdaParam {
    pub name: InternedString,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedLambdaBody {
    Expression(Box<TypedNode<TypedExpression>>),
    Block(TypedBlock),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedCapture {
    pub name: InternedString,
    pub ty: Type,
    pub by_ref: bool,
    pub mutability: Mutability,
}

/// Match expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMatchExpr {
    pub scrutinee: Box<TypedNode<TypedExpression>>,
    pub arms: Vec<TypedMatchArm>,
}

/// If expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedIfExpr {
    pub condition: Box<TypedNode<TypedExpression>>,
    pub then_branch: Box<TypedNode<TypedExpression>>,
    pub else_branch: Box<TypedNode<TypedExpression>>,
}

/// Type cast
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedCast {
    pub expr: Box<TypedNode<TypedExpression>>,
    pub target_type: Type,
}

/// Reference expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedReference {
    pub expr: Box<TypedNode<TypedExpression>>,
    pub mutability: Mutability,
}

/// Range expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedRange {
    pub start: Option<Box<TypedNode<TypedExpression>>>,
    pub end: Option<Box<TypedNode<TypedExpression>>>,
    pub inclusive: bool,
}

/// List comprehension: [expr for var in iter if condition]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedListComprehension {
    /// The output expression evaluated for each element
    pub output_expr: Box<TypedNode<TypedExpression>>,
    /// The loop variable name
    pub variable: InternedString,
    /// The iterable expression
    pub iterator: Box<TypedNode<TypedExpression>>,
    /// Optional filter condition
    pub condition: Option<Box<TypedNode<TypedExpression>>>,
}

/// Slice expression: arr[start:end:step]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedSlice {
    /// The object being sliced
    pub object: Box<TypedNode<TypedExpression>>,
    /// Start index (None = beginning)
    pub start: Option<Box<TypedNode<TypedExpression>>>,
    /// End index (None = end)
    pub end: Option<Box<TypedNode<TypedExpression>>>,
    /// Step value (None = 1)
    pub step: Option<Box<TypedNode<TypedExpression>>>,
}

/// Import modifier expression: import loader("path") as Type
///
/// Loads an asset file using a specified loader function and returns it
/// as an opaque type. The loader (e.g., `asset`, `image`, `audio`, `model`)
/// determines how the file is loaded, and the target type specifies the
/// expected return type.
///
/// Example: `import asset("image.jpg") as Image`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedImportModifier {
    /// The loader function name (e.g., "asset", "image", "audio")
    pub loader: InternedString,
    /// Path to the asset file
    pub path: InternedString,
    /// Expected return type name
    pub target_type: InternedString,
}

/// Path expression: Type::method or module::function
///
/// Used for associated functions (static methods) and qualified paths.
///
/// Example: `Tensor::zeros`, `std::Option::Some`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPath {
    /// Path segments (e.g., ["Tensor", "zeros"] for Tensor::zeros)
    pub segments: Vec<InternedString>,
}

/// Method call with enhanced argument support
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMethodCall {
    pub receiver: Box<TypedNode<TypedExpression>>,
    pub method: InternedString,
    pub type_args: Vec<Type>,
    pub positional_args: Vec<TypedNode<TypedExpression>>,
    pub named_args: Vec<TypedNamedArg>,
}

/// For loop (iterator-style: for item in collection)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedFor {
    pub pattern: Box<TypedNode<TypedPattern>>,
    pub iterator: Box<TypedNode<TypedExpression>>,
    pub body: TypedBlock,
}

/// C-style for loop (for init; condition; update)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedForCStyle {
    pub init: Option<Box<TypedNode<TypedStatement>>>,
    pub condition: Option<Box<TypedNode<TypedExpression>>>,
    pub update: Option<Box<TypedNode<TypedExpression>>>,
    pub body: TypedBlock,
    pub span: Span,
}

/// Loop construct - unified representation for various loop types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedLoop {
    /// Iterator-based: for item in collection
    ForEach {
        pattern: Box<TypedNode<TypedPattern>>,
        iterator: Box<TypedNode<TypedExpression>>,
        body: TypedBlock,
    },
    /// C-style: for (init; condition; update)
    ForCStyle {
        init: Option<Box<TypedNode<TypedStatement>>>,
        condition: Option<Box<TypedNode<TypedExpression>>>,
        update: Option<Box<TypedNode<TypedExpression>>>,
        body: TypedBlock,
    },
    /// While: while condition
    While {
        condition: Box<TypedNode<TypedExpression>>,
        body: TypedBlock,
    },
    /// Do-while: do body while condition
    DoWhile {
        body: TypedBlock,
        condition: Box<TypedNode<TypedExpression>>,
    },
    /// Infinite loop: loop { ... } (Rust-style)
    Infinite { body: TypedBlock },
}

/// Match/Switch statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMatch {
    pub scrutinee: Box<TypedNode<TypedExpression>>,
    pub arms: Vec<TypedMatchArm>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMatchArm {
    pub pattern: Box<TypedNode<TypedPattern>>,
    pub guard: Option<Box<TypedNode<TypedExpression>>>,
    pub body: Box<TypedNode<TypedExpression>>,
}

/// Try/Catch
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedTry {
    pub body: TypedBlock,
    pub catch_clauses: Vec<TypedCatch>,
    pub finally_block: Option<TypedBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedCatch {
    pub pattern: Box<TypedNode<TypedPattern>>,
    pub body: TypedBlock,
}

/// Pattern matching - Advanced system supporting Rust/Haxe-style patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum TypedPattern {
    /// Wildcard pattern: _
    #[default]
    Wildcard,

    /// Variable binding: x, mut y
    Identifier {
        name: InternedString,
        mutability: Mutability,
    },

    /// Literal patterns: 42, "hello", true
    Literal(TypedLiteralPattern),

    /// Tuple patterns: (x, y, _)
    Tuple(Vec<TypedNode<TypedPattern>>),

    /// Struct patterns: Point { x, y } or Point { x: px, y: py }
    Struct {
        name: InternedString,
        fields: Vec<TypedFieldPattern>,
    },

    /// Enum variant patterns: Some(x), Ok(value), None
    Enum {
        name: InternedString,
        variant: InternedString,
        fields: Vec<TypedNode<TypedPattern>>,
    },

    /// Array/Vec patterns: [x, y, z]
    Array(Vec<TypedNode<TypedPattern>>),

    /// Slice patterns: [head, tail @ ..]
    Slice {
        prefix: Vec<TypedNode<TypedPattern>>,
        middle: Option<Box<TypedNode<TypedPattern>>>, // rest pattern: ..rest
        suffix: Vec<TypedNode<TypedPattern>>,
    },

    /// Range patterns: 1..=10, 'a'..='z'
    Range {
        start: Box<TypedNode<TypedLiteralPattern>>,
        end: Box<TypedNode<TypedLiteralPattern>>,
        inclusive: bool,
    },

    /// Reference patterns: &x, &mut y
    Reference {
        pattern: Box<TypedNode<TypedPattern>>,
        mutability: Mutability,
    },

    /// Box/Pointer patterns: Box(x)
    Box(Box<TypedNode<TypedPattern>>),

    /// Or patterns: x | y | z
    Or(Vec<TypedNode<TypedPattern>>),

    /// Guard patterns: x if x > 0
    Guard {
        pattern: Box<TypedNode<TypedPattern>>,
        condition: Box<TypedNode<TypedExpression>>,
    },

    /// Rest patterns: ..rest (in tuples/arrays)
    Rest {
        name: Option<InternedString>,
        mutability: Mutability,
    },

    /// At patterns: binding @ pattern (e.g., x @ Some(y))
    At {
        name: InternedString,
        mutability: Mutability,
        pattern: Box<TypedNode<TypedPattern>>,
    },

    /// Constant patterns: CONST_VALUE
    Constant(InternedString),

    /// Path patterns: std::Some, MyEnum::Variant
    Path {
        path: Vec<InternedString>,
        args: Option<Vec<TypedNode<TypedPattern>>>,
    },

    /// Regex patterns (Haxe-style): ~/pattern/flags
    Regex {
        pattern: InternedString,
        flags: Option<InternedString>,
    },

    /// Type patterns: x: T (explicit type annotation in pattern)
    Typed {
        pattern: Box<TypedNode<TypedPattern>>,
        ty: Type,
    },

    /// Macro patterns: pattern!(...) for language-specific extensions
    Macro {
        name: InternedString,
        args: Vec<TypedNode<TypedPattern>>,
    },

    /// Map/Object patterns: { key1: p1, key2: p2, ..rest }
    Map(TypedMapPattern),

    /// View patterns (active patterns): view_func => pattern
    View {
        view_function: Box<TypedNode<TypedExpression>>,
        pattern: Box<TypedNode<TypedPattern>>,
    },

    /// Lazy patterns: lazy(pattern) - for lazy evaluation
    Lazy(Box<TypedNode<TypedPattern>>),

    /// When patterns: pattern when condition (F#/ML style)
    When {
        pattern: Box<TypedNode<TypedPattern>>,
        condition: Box<TypedNode<TypedExpression>>,
    },

    /// Async patterns: async pattern (for async/await contexts)
    Async(Box<TypedNode<TypedPattern>>),

    /// Constructor patterns with type args: Vec<T>(pattern)
    Constructor {
        constructor: Type,
        pattern: Box<TypedNode<TypedPattern>>,
    },

    /// Error patterns: Error(kind, message) for exception handling
    Error {
        error_type: Option<Type>,
        pattern: Box<TypedNode<TypedPattern>>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedFieldPattern {
    pub name: InternedString,
    pub pattern: Box<TypedNode<TypedPattern>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedLiteralPattern {
    /// Integer literals: 42, -10, 0xFF
    Integer(i128),
    /// Float literals: 3.14, -2.5
    Float(f64),
    /// Boolean literals: true, false
    Bool(bool),
    /// String literals: "hello", r"raw string"
    String(InternedString),
    /// Character literals: 'a', '\n'
    Char(char),
    /// Byte literals: b'A' (for Rust)
    Byte(u8),
    /// Byte string literals: b"hello" (for Rust)
    ByteString(Vec<u8>),
    /// Unit literal: ()
    Unit,
    /// Null/None/nil literals
    Null,
}

/// Additional pattern-related types for advanced matching

/// Pattern destructuring for map/object patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMapPattern {
    pub entries: Vec<TypedMapPatternEntry>,
    pub exhaustive: bool, // true if all keys must be matched
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedMapPatternEntry {
    /// Key-value pattern: { key: pattern }
    KeyValue {
        key: TypedNode<TypedLiteralPattern>,
        pattern: Box<TypedNode<TypedPattern>>,
    },
    /// Rest pattern: { ..rest }
    Rest {
        name: Option<InternedString>,
        mutability: Mutability,
    },
}

/// Pattern guards for conditional matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPatternGuard {
    pub condition: Box<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// Pattern alternatives (or-patterns with different bindings)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedPatternAlternative {
    pub patterns: Vec<TypedNode<TypedPattern>>,
    pub guard: Option<TypedPatternGuard>,
}

/// Helper methods for pattern construction
impl TypedPattern {
    /// Create a wildcard pattern
    pub fn wildcard() -> Self {
        TypedPattern::Wildcard
    }

    /// Create a variable binding pattern
    pub fn var(name: InternedString, mutability: Mutability) -> Self {
        TypedPattern::Identifier { name, mutability }
    }

    /// Create an immutable variable binding
    pub fn immutable_var(name: InternedString) -> Self {
        TypedPattern::Identifier {
            name,
            mutability: Mutability::Immutable,
        }
    }

    /// Create a mutable variable binding
    pub fn mutable_var(name: InternedString) -> Self {
        TypedPattern::Identifier {
            name,
            mutability: Mutability::Mutable,
        }
    }

    /// Create a literal pattern
    pub fn literal(lit: TypedLiteralPattern) -> Self {
        TypedPattern::Literal(lit)
    }

    /// Create an integer literal pattern
    pub fn int(value: i128) -> Self {
        TypedPattern::Literal(TypedLiteralPattern::Integer(value))
    }

    /// Create a string literal pattern
    pub fn string(value: InternedString) -> Self {
        TypedPattern::Literal(TypedLiteralPattern::String(value))
    }

    /// Create a boolean literal pattern
    pub fn bool(value: bool) -> Self {
        TypedPattern::Literal(TypedLiteralPattern::Bool(value))
    }

    /// Create a tuple pattern
    pub fn tuple(patterns: Vec<TypedNode<TypedPattern>>) -> Self {
        TypedPattern::Tuple(patterns)
    }

    /// Create an or-pattern (alternative patterns)
    pub fn or(patterns: Vec<TypedNode<TypedPattern>>) -> Self {
        TypedPattern::Or(patterns)
    }

    /// Create a guard pattern
    pub fn guard(pattern: TypedNode<TypedPattern>, condition: TypedNode<TypedExpression>) -> Self {
        TypedPattern::Guard {
            pattern: Box::new(pattern),
            condition: Box::new(condition),
        }
    }

    /// Create a range pattern
    pub fn range(start: TypedLiteralPattern, end: TypedLiteralPattern, inclusive: bool) -> Self {
        TypedPattern::Range {
            start: Box::new(typed_node(start, Type::Never, Span::new(0, 0))),
            end: Box::new(typed_node(end, Type::Never, Span::new(0, 0))),
            inclusive,
        }
    }

    /// Create an at-pattern (binding @ pattern)
    pub fn at(
        name: InternedString,
        mutability: Mutability,
        pattern: TypedNode<TypedPattern>,
    ) -> Self {
        TypedPattern::At {
            name,
            mutability,
            pattern: Box::new(pattern),
        }
    }

    /// Create a reference pattern
    pub fn reference(pattern: TypedNode<TypedPattern>, mutability: Mutability) -> Self {
        TypedPattern::Reference {
            pattern: Box::new(pattern),
            mutability,
        }
    }

    /// Check if pattern binds any variables
    pub fn binds_variables(&self) -> bool {
        match self {
            TypedPattern::Wildcard | TypedPattern::Literal(_) | TypedPattern::Constant(_) => false,
            TypedPattern::Identifier { .. } | TypedPattern::Rest { name: Some(_), .. } => true,
            TypedPattern::Tuple(patterns) => patterns.iter().any(|p| p.node.binds_variables()),
            TypedPattern::Array(patterns) => patterns.iter().any(|p| p.node.binds_variables()),
            TypedPattern::Or(patterns) => patterns.iter().any(|p| p.node.binds_variables()),
            TypedPattern::At { .. } => true,
            TypedPattern::Reference { pattern, .. } => pattern.node.binds_variables(),
            TypedPattern::Box(pattern) => pattern.node.binds_variables(),
            TypedPattern::Guard { pattern, .. } => pattern.node.binds_variables(),
            TypedPattern::Struct { fields, .. } => {
                fields.iter().any(|f| f.pattern.node.binds_variables())
            }
            TypedPattern::Enum { fields, .. } => fields.iter().any(|f| f.node.binds_variables()),
            _ => true, // Conservative approach for other patterns
        }
    }

    /// Check if pattern is exhaustive (matches all possible values)
    pub fn is_exhaustive(&self) -> bool {
        matches!(
            self,
            TypedPattern::Wildcard | TypedPattern::Identifier { .. }
        )
    }
}

// ====== EXTENDED TYPE DEFINITIONS ======
// Adding more language features incrementally

/// Class declaration (for OOP languages)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedClass {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub extends: Option<Type>,
    pub implements: Vec<Type>,
    pub fields: Vec<TypedField>,
    pub methods: Vec<TypedMethod>,
    pub constructors: Vec<TypedConstructor>,
    pub visibility: Visibility,
    pub is_abstract: bool,
    pub is_final: bool,
    pub span: Span,
}

/// Interface/Trait declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedInterface {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub extends: Vec<Type>,
    pub methods: Vec<TypedMethodSignature>,
    pub associated_types: Vec<TypedAssociatedType>,
    pub visibility: Visibility,
    pub span: Span,
}

/// Trait implementation block
/// Represents: impl TraitName<TypeArgs> for TypeName { ... }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedTraitImpl {
    /// The trait being implemented (e.g., "Add" in "impl Add<T> for Vec<T>")
    pub trait_name: InternedString,
    /// Type arguments for the trait (e.g., <Tensor> in "impl Add<Tensor> for Tensor")
    pub trait_type_args: Vec<Type>,
    /// The type implementing the trait (e.g., "Tensor" in "impl Add for Tensor")
    pub for_type: Type,
    /// Method implementations
    pub methods: Vec<TypedMethod>,
    /// Associated type definitions (e.g., "type Output = Tensor")
    pub associated_types: Vec<TypedImplAssociatedType>,
    pub span: Span,
}

/// Associated type definition in impl block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedImplAssociatedType {
    pub name: InternedString,
    pub ty: Type,
    pub span: Span,
}

/// Enum declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedEnum {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub variants: Vec<TypedVariant>,
    pub visibility: Visibility,
    pub span: Span,
}

/// Type alias declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedTypeAlias {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub target: Type,
    pub visibility: Visibility,
    pub span: Span,
}

/// Module/Namespace declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedModule {
    pub name: InternedString,
    pub declarations: Vec<TypedNode<TypedDeclaration>>,
    pub visibility: Visibility,
    pub span: Span,
}

/// Import declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedImport {
    pub module_path: Vec<InternedString>,
    pub items: Vec<TypedImportItem>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedImportItem {
    Named {
        name: InternedString,
        alias: Option<InternedString>,
    },
    Glob,
    Default(InternedString),
}

/// Type parameter with bounds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedTypeParam {
    pub name: InternedString,
    pub bounds: Vec<TypedTypeBound>,
    pub default: Option<Type>,
    pub span: Span,
}

/// Type bounds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedTypeBound {
    /// Trait/Interface bound: T: Display
    Trait(Type),
    /// Lifetime bound: T: 'a
    Lifetime(crate::type_registry::Lifetime),
    /// Equality constraint: T = U (associated types)
    Equality(Type),
    /// Subtype constraint: T <: U (Java/C# style)
    Subtype(Type),
    /// Supertype constraint: T :> U (Scala style)  
    Supertype(Type),
    /// Constructor constraint: T: new() (C# style)
    Constructor(Vec<Type>),
    /// Size constraint: T: Sized (Rust style)
    Sized,
    /// Copy constraint: T: Copy (Rust style)
    Copy,
    /// Send constraint: T: Send (Rust style)
    Send,
    /// Sync constraint: T: Sync (Rust style)
    Sync,
    /// Static lifetime constraint: T: 'static
    Static,
    /// Value type constraint (C# struct)
    ValueType,
    /// Reference type constraint (C# class)
    ReferenceType,
    /// Unmanaged constraint (C#)
    Unmanaged,
    /// Custom constraint with parameters
    Custom {
        name: InternedString,
        args: Vec<Type>,
    },
}

/// Class/Struct field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedField {
    pub name: InternedString,
    pub ty: Type,
    pub initializer: Option<Box<TypedNode<TypedExpression>>>,
    pub visibility: Visibility,
    pub mutability: Mutability,
    pub is_static: bool,
    pub span: Span,
}

/// Method implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMethod {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub params: Vec<TypedMethodParam>,
    pub return_type: Type,
    pub body: Option<TypedBlock>,
    pub visibility: Visibility,
    pub is_static: bool,
    pub is_async: bool,
    pub is_override: bool,
    pub span: Span,
}

/// Method parameter (includes self) with enhanced features
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMethodParam {
    pub name: InternedString,
    pub ty: Type,
    pub mutability: Mutability,
    pub is_self: bool,
    pub kind: ParameterKind,
    pub default_value: Option<Box<TypedNode<TypedExpression>>>,
    pub attributes: Vec<ParameterAttribute>,
    pub span: Span,
}

/// Constructor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedConstructor {
    pub params: Vec<TypedMethodParam>,
    pub body: TypedBlock,
    pub visibility: Visibility,
    pub span: Span,
}

/// Method signature (for interfaces)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedMethodSignature {
    pub name: InternedString,
    pub type_params: Vec<TypedTypeParam>,
    pub params: Vec<TypedMethodParam>,
    pub return_type: Type,
    pub is_static: bool,
    pub is_async: bool,
    pub span: Span,
}

/// Associated type in interface/trait
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedAssociatedType {
    pub name: InternedString,
    pub bounds: Vec<TypedTypeBound>,
    pub default: Option<Type>,
    pub span: Span,
}

/// Enum variant
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedVariant {
    pub name: InternedString,
    pub fields: TypedVariantFields,
    pub discriminant: Option<Box<TypedNode<TypedExpression>>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedVariantFields {
    Unit,
    Tuple(Vec<Type>),
    Named(Vec<TypedField>),
}

// ====== COROUTINE AND ASYNC STATEMENTS ======

/// Coroutine statement - general async/coroutine construct
/// Supports Go goroutines, async/await, yield patterns, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedCoroutine {
    /// Type of coroutine: Goroutine, Async, Generator, etc.
    pub kind: CoroutineKind,
    /// The expression or block to execute as a coroutine
    pub body: Box<TypedNode<TypedExpression>>,
    /// Optional parameters passed to the coroutine
    pub params: Vec<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// Different types of coroutines supported across languages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoroutineKind {
    /// Go-style goroutine: go func() { ... }()
    Goroutine,
    /// Rust/JS-style async: async { ... }
    Async,
    /// Python/C#-style generator: yield expression
    Generator,
    /// Custom coroutine with specific semantics
    Custom { name: InternedString },
}

/// Defer statement - for stack-based cleanup (Go, Swift, etc.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedDefer {
    /// Expression to execute on scope exit
    pub body: Box<TypedNode<TypedExpression>>,
    pub span: Span,
}

/// Select statement - for channel operations (Go, Rust async)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedSelect {
    /// List of select arms
    pub arms: Vec<TypedSelectArm>,
    /// Optional default case
    pub default: Option<TypedBlock>,
    pub span: Span,
}

/// Select arm - individual case in select statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedSelectArm {
    /// Operation: send, receive, or timeout
    pub operation: TypedSelectOperation,
    /// Code to execute if this arm is selected
    pub body: TypedBlock,
    pub span: Span,
}

/// Select operations for different communication patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypedSelectOperation {
    /// Receive from channel: <-ch or ch.recv()
    Receive {
        channel: Box<TypedNode<TypedExpression>>,
        pattern: Option<Box<TypedNode<TypedPattern>>>,
    },
    /// Send to channel: ch <- value or ch.send(value)
    Send {
        channel: Box<TypedNode<TypedExpression>>,
        value: Box<TypedNode<TypedExpression>>,
    },
    /// Timeout operation: after duration
    Timeout {
        duration: Box<TypedNode<TypedExpression>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_registry::PrimitiveType;

    #[test]
    fn test_typed_node_creation() {
        let span = Span::new(0, 10);
        let expr = TypedExpression::Literal(TypedLiteral::Integer(42));
        let node = typed_node(expr, Type::Primitive(PrimitiveType::I32), span);

        assert_eq!(node.span, span);
        assert!(matches!(node.ty, Type::Primitive(PrimitiveType::I32)));
    }

    #[test]
    fn test_variable_mutability() {
        let mut arena = crate::arena::AstArena::new();
        let name = arena.intern_string("test");
        let span = Span::new(0, 10);
        let let_stmt = TypedLet {
            name,
            ty: Type::Primitive(PrimitiveType::I32),
            mutability: Mutability::Mutable,
            initializer: None,
            span,
        };

        assert_eq!(let_stmt.mutability, Mutability::Mutable);
        assert_eq!(let_stmt.span, span);
    }
}
