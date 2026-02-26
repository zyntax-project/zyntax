//! # AST Conversion Traits
//!
//! Traits and utilities for converting from language-specific ASTs
//! to the Zyntax TypedAST without creating intermediate representations.
//!
//! ## Design Philosophy
//! - Language developers already have their own AST
//! - Direct conversion avoids unnecessary allocations
//! - Type information is computed during conversion
//! - Source locations are preserved from the original AST

use crate::arena::InternedString;
use crate::source::Span;
use crate::type_registry::{Mutability, Type, Visibility};
use crate::typed_ast::{
    ParameterKind, TypedDeclaration, TypedExpression, TypedNode, TypedProgram, TypedStatement,
};
// All types are now in typed_ast module
use std::error::Error;
use std::fmt;

/// Error during AST conversion
#[derive(Debug)]
pub enum ConversionError {
    /// Type cannot be resolved
    UnresolvedType(String),
    /// Invalid AST structure
    InvalidStructure(String),
    /// Missing required information
    MissingInfo(String),
    /// Custom error from language implementation
    Custom(Box<dyn Error + Send + Sync>),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::UnresolvedType(msg) => write!(f, "Unresolved type: {}", msg),
            ConversionError::InvalidStructure(msg) => write!(f, "Invalid structure: {}", msg),
            ConversionError::MissingInfo(msg) => write!(f, "Missing information: {}", msg),
            ConversionError::Custom(err) => write!(f, "Conversion error: {}", err),
        }
    }
}

impl Error for ConversionError {}

/// Result type for conversions
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Context for AST conversion
pub struct ConversionContext {
    /// Type registry for resolving types  
    pub type_registry: Box<crate::type_registry::TypeRegistry>,
    /// String interner for efficient string handling
    pub strings: Box<dyn StringInterner>,
    /// Current module/namespace path
    pub module_path: Vec<InternedString>,
    /// Type resolution cache
    type_cache: std::collections::HashMap<String, Type>,
}

/// String interner interface for conversion
pub trait StringInterner {
    fn intern(&mut self, s: &str) -> InternedString;
}

impl ConversionContext {
    pub fn new(
        type_registry: Box<crate::type_registry::TypeRegistry>,
        strings: Box<dyn StringInterner>,
    ) -> Self {
        Self {
            type_registry,
            strings,
            module_path: Vec::new(),
            type_cache: std::collections::HashMap::new(),
        }
    }

    /// Enter a module scope
    pub fn enter_module(&mut self, name: InternedString) {
        self.module_path.push(name);
    }

    /// Leave a module scope
    pub fn leave_module(&mut self) {
        self.module_path.pop();
    }

    /// Cache a resolved type
    pub fn cache_type(&mut self, key: String, ty: Type) {
        self.type_cache.insert(key, ty);
    }

    /// Get a cached type
    pub fn get_cached_type(&self, key: &str) -> Option<&Type> {
        self.type_cache.get(key)
    }
}

// Wrapper to make any string interner work with our trait
struct StringInternerWrapper(Box<dyn StringInterner>);

impl StringInterner for StringInternerWrapper {
    fn intern(&mut self, s: &str) -> InternedString {
        self.0.intern(s)
    }
}

/// Trait for converting language-specific AST nodes to TypedAST
pub trait ToTypedAst<T> {
    /// The source AST type
    type Source;

    /// Convert to TypedAST node with type and span
    fn to_typed_ast(
        &self,
        source: &Self::Source,
        ctx: &mut ConversionContext,
    ) -> ConversionResult<T>;
}

/// Trait for converting programs
pub trait ProgramConverter {
    type Program;

    fn convert_program(
        &self,
        program: &Self::Program,
        ctx: &mut ConversionContext,
    ) -> ConversionResult<TypedProgram>;
}

/// Trait for converting declarations
pub trait DeclarationConverter {
    type Declaration;

    fn convert_declaration(
        &self,
        decl: &Self::Declaration,
        ctx: &mut ConversionContext,
    ) -> ConversionResult<TypedNode<TypedDeclaration>>;
}

/// Trait for converting statements
pub trait StatementConverter {
    type Statement;

    fn convert_statement(
        &self,
        stmt: &Self::Statement,
        ctx: &mut ConversionContext,
    ) -> ConversionResult<TypedNode<TypedStatement>>;
}

/// Trait for converting expressions
pub trait ExpressionConverter {
    type Expression;

    fn convert_expression(
        &self,
        expr: &Self::Expression,
        ctx: &mut ConversionContext,
    ) -> ConversionResult<TypedNode<TypedExpression>>;
}

/// Trait for converting types
pub trait TypeConverter {
    type Type;

    fn convert_type(&self, ty: &Self::Type, ctx: &mut ConversionContext) -> ConversionResult<Type>;
}

/// Trait for extracting source location information
pub trait SourceLocation {
    fn span(&self) -> Span;
}

/// Helper trait for common conversions
pub trait ConversionHelpers {
    /// Convert visibility modifier
    fn convert_visibility(&self, vis: &str) -> Visibility {
        match vis {
            "public" => Visibility::Public,
            "private" => Visibility::Private,
            "protected" => Visibility::Protected,
            "internal" => Visibility::Internal,
            _ => Visibility::Private,
        }
    }

    /// Convert mutability
    fn convert_mutability(&self, is_mutable: bool) -> Mutability {
        if is_mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        }
    }
}

/// Example implementation for a hypothetical TypeScript AST
pub mod typescript_example {
    use super::*;

    // Hypothetical TypeScript AST types
    pub struct TSProgram {
        pub statements: Vec<TSStatement>,
    }

    pub enum TSStatement {
        Function {
            name: String,
            params: Vec<TSParam>,
            body: TSBlock,
        },
        Variable {
            name: String,
            ty: Option<TSType>,
            init: Option<TSExpression>,
        },
    }

    pub struct TSParam {
        pub name: String,
        pub ty: TSType,
    }

    pub struct TSType {
        pub kind: String,
    }

    pub struct TSBlock {
        pub statements: Vec<TSStatement>,
    }

    pub struct TSExpression {
        pub kind: String,
    }

    // Converter implementation
    pub struct TypeScriptConverter;

    impl ProgramConverter for TypeScriptConverter {
        type Program = TSProgram;

        fn convert_program(
            &self,
            program: &TSProgram,
            ctx: &mut ConversionContext,
        ) -> ConversionResult<TypedProgram> {
            let mut declarations = Vec::new();

            for stmt in &program.statements {
                // Convert each statement to declaration
                // This is where the actual conversion logic would go
            }

            Ok(TypedProgram {
                declarations,
                span: Span::new(0, 0), // Would get from source
                source_files: vec![],  // TODO: Add source file info
                type_registry: crate::TypeRegistry::new(),
            })
        }
    }
}

/// Builder pattern for direct AST construction
pub struct TypedAstBuilder<'a> {
    ctx: &'a mut ConversionContext,
}

impl<'a> TypedAstBuilder<'a> {
    pub fn new(ctx: &'a mut ConversionContext) -> Self {
        Self { ctx }
    }

    /// Build a typed node
    pub fn typed_node<T>(&mut self, node: T, ty: Type, span: Span) -> TypedNode<T> {
        TypedNode::new(node, ty, span)
    }

    /// Intern a string
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.ctx.strings.intern(s)
    }

    /// Build a function declaration
    pub fn function(
        &mut self,
        name: &str,
        params: Vec<(String, Type, Mutability)>,
        return_type: Type,
        body: crate::typed_ast::TypedBlock,
        visibility: Visibility,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        let name = self.intern(name);
        let params = params
            .into_iter()
            .map(|(n, t, m)| crate::typed_ast::TypedParameter {
                name: self.intern(&n),
                ty: t,
                mutability: m,
                span: span.clone(),
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
            })
            .collect();

        let func = crate::typed_ast::TypedFunction {
            name,
            annotations: vec![],
            effects: vec![],
            type_params: vec![],
            params,
            return_type: return_type.clone(),
            body: Some(body),
            visibility,
            is_async: false,
            is_pure: false,
            is_external: false,
            calling_convention: crate::type_registry::CallingConvention::Default,
            link_name: None,
        };

        self.typed_node(TypedDeclaration::Function(func), return_type, span)
    }

    /// Build a variable declaration
    pub fn variable(
        &mut self,
        name: &str,
        ty: Type,
        mutability: Mutability,
        initializer: Option<Box<TypedNode<TypedExpression>>>,
        visibility: Visibility,
        span: Span,
    ) -> TypedNode<TypedDeclaration> {
        let name = self.intern(name);

        let var = crate::typed_ast::TypedVariable {
            name,
            ty: ty.clone(),
            mutability,
            initializer,
            visibility,
        };

        self.typed_node(TypedDeclaration::Variable(var), ty, span)
    }
}
