//! # Language-Agnostic Type Registry
//!
//! This module provides a flexible type registry for built-in complex types
//! like `List<T>`, `HashMap<K,V>`, `Set<T>`, etc. It complements the basic
//! `PrimitiveType` enum, which handles simple atomic types.
//!
//! ## Purpose
//!
//! - **Primitive types** (int, bool, string): Use `Type::Primitive(PrimitiveType::*)`
//! - **Complex built-in types** (List, Map, etc.): Use `Type::Named` with TypeRegistry
//! - **User-defined types** (classes, structs): Also use TypeRegistry
//!
//! This separation allows language implementers to:
//! 1. Use universal primitive types without registration
//! 2. Register their language's specific complex built-in types
//! 3. Maintain full control over type system without hardcoded assumptions

use crate::arena::InternedString;
use crate::source::Span;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

/// Global type ID counter for generating unique type identifiers
static NEXT_TYPE_ID: AtomicU32 = AtomicU32::new(1);

/// Unique identifier for registered types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeId(u32);

impl TypeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn next() -> Self {
        Self(NEXT_TYPE_ID.fetch_add(1, Ordering::SeqCst))
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Primitive types common across languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveType {
    // Numeric types
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,

    // Other primitives
    Bool,
    Char,
    String,

    // Platform-specific integers
    ISize,
    USize,

    // Special
    Unit, // void/unit type
}

impl PrimitiveType {
    /// Check if this is a numeric type
    pub fn is_numeric(self) -> bool {
        matches!(
            self,
            PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::I128
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64
                | PrimitiveType::U128
                | PrimitiveType::F32
                | PrimitiveType::F64
                | PrimitiveType::ISize
                | PrimitiveType::USize
        )
    }

    /// Check if this is an integral type
    pub fn is_integral(self) -> bool {
        matches!(
            self,
            PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::I128
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64
                | PrimitiveType::U128
                | PrimitiveType::ISize
                | PrimitiveType::USize
        )
    }

    /// Check if this is a floating point type
    pub fn is_float(self) -> bool {
        matches!(self, PrimitiveType::F32 | PrimitiveType::F64)
    }

    /// Get the size of this type in bytes (if known)
    pub fn size_of(self) -> Option<usize> {
        match self {
            PrimitiveType::I8 | PrimitiveType::U8 => Some(1),
            PrimitiveType::I16 | PrimitiveType::U16 => Some(2),
            PrimitiveType::I32 | PrimitiveType::U32 | PrimitiveType::F32 => Some(4),
            PrimitiveType::I64 | PrimitiveType::U64 | PrimitiveType::F64 => Some(8),
            PrimitiveType::I128 | PrimitiveType::U128 => Some(16),
            PrimitiveType::Bool => Some(1),
            PrimitiveType::Char => Some(4), // UTF-32
            PrimitiveType::Unit => Some(0),
            // Platform and implementation dependent
            PrimitiveType::String | PrimitiveType::ISize | PrimitiveType::USize => None,
        }
    }
}

/// Nullability kinds for enhanced null safety
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NullabilityKind {
    /// Explicitly non-null
    NonNull,
    /// Explicitly nullable
    Nullable,
    /// Nullability unknown (for gradual typing)
    Unknown,
    /// Platform-dependent nullability
    Platform,
}

/// Async kinds for different asynchronous programming models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AsyncKind {
    /// Synchronous execution
    Sync,
    /// Basic async/await
    Async,
    /// Future/Promise
    Future,
    /// Task-based
    Task,
    /// Promise-based (JavaScript style)
    Promise,
    /// Coroutine-based
    Coroutine,
    /// Generator/yield
    Generator,
}

/// Calling conventions for function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallingConvention {
    /// Default platform calling convention
    Default,
    /// C calling convention
    Cdecl,
    /// Standard call (Windows)
    Stdcall,
    /// Fast call
    Fastcall,
    /// This call (C++)
    Thiscall,
    /// Vector call
    Vectorcall,
    /// Rust calling convention
    Rust,
    /// System calling convention
    System,
}

/// Const value representation for const generics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstValue {
    /// Integer constant
    Int(i64),
    /// Unsigned integer constant
    UInt(u64),
    /// Boolean constant
    Bool(bool),
    /// String constant
    String(InternedString),
    /// Character constant
    Char(char),
    /// Array constant
    Array(Vec<ConstValue>),
    /// Tuple constant
    Tuple(Vec<ConstValue>),
    /// Struct constant
    Struct(Vec<(InternedString, ConstValue)>),
    /// Variable reference
    Variable(InternedString),
    /// Function call
    FunctionCall {
        name: InternedString,
        args: Vec<ConstValue>,
    },
    /// Binary operation
    BinaryOp {
        op: ConstBinaryOp,
        left: Box<ConstValue>,
        right: Box<ConstValue>,
    },
    /// Unary operation
    UnaryOp {
        op: ConstUnaryOp,
        operand: Box<ConstValue>,
    },
}

/// Binary operators for const expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogicalAnd,
    LogicalOr,
}

/// Unary operators for const expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstUnaryOp {
    Neg,
    Not,
    LogicalNot,
}

/// Const constraint for const-dependent types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstConstraint {
    /// Value must equal a specific const
    Equal(ConstValue),
    /// Value must be in a range
    Range { min: ConstValue, max: ConstValue },
    /// Value must satisfy a predicate
    Predicate(ConstPredicate),
    /// Multiple constraints (all must hold)
    And(Vec<ConstConstraint>),
    /// Alternative constraints (at least one must hold)
    Or(Vec<ConstConstraint>),
}

/// Const predicate for refinement types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstPredicate {
    pub name: InternedString,
    pub args: Vec<ConstValue>,
}

/// Unique identifier for const variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstVarId(u32);

impl ConstVarId {
    pub fn next() -> Self {
        static NEXT_CONST_VAR: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_CONST_VAR.fetch_add(1, Ordering::SeqCst))
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Kind system for higher-kinded types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// Type kind (*)
    Type,
    /// Function kind (* -> *)
    Arrow { from: Box<Kind>, to: Box<Kind> },
    /// Constraint kind
    Constraint,
}

impl Default for NullabilityKind {
    fn default() -> Self {
        NullabilityKind::Unknown
    }
}

impl Default for AsyncKind {
    fn default() -> Self {
        AsyncKind::Sync
    }
}

impl Default for CallingConvention {
    fn default() -> Self {
        CallingConvention::Default
    }
}

/// Core type representation that is language-agnostic and supports all paradigms
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Type {
    /// Primitive types (int, float, bool, etc.)
    Primitive(PrimitiveType),

    /// The bottom type (never/empty) - default for uninitialized types
    #[default]
    Never,

    /// A type registered in the type registry by ID
    Named {
        id: TypeId,
        type_args: Vec<Type>,
        const_args: Vec<ConstValue>,
        variance: Vec<Variance>,
        nullability: NullabilityKind,
    },

    /// Type variable for inference (e.g., T, U)
    TypeVar(TypeVar),

    /// Function type: (params) -> return_type with enhanced parameter info
    Function {
        params: Vec<ParamInfo>,
        return_type: Box<Type>,
        is_varargs: bool,
        has_named_params: bool,
        has_default_params: bool,
        async_kind: AsyncKind,
        calling_convention: CallingConvention,
        nullability: NullabilityKind,
    },

    /// Tuple type: (T1, T2, ..., Tn)
    Tuple(Vec<Type>),

    /// Array type: [T; size] or T[]
    Array {
        element_type: Box<Type>,
        size: Option<ConstValue>, // None for dynamic arrays, ConstValue for const generics
        nullability: NullabilityKind,
    },

    /// Reference/Pointer type
    Reference {
        ty: Box<Type>,
        mutability: Mutability,
        lifetime: Option<Lifetime>,
        nullability: NullabilityKind,
    },

    /// Optional/Nullable type: T? or Option<T>
    Optional(Box<Type>),

    /// Result/Error union type: Result<T, E> or !T (Zig)
    Result {
        ok_type: Box<Type>,
        err_type: Box<Type>,
    },

    /// Union type: T1 | T2 | ... | Tn
    Union(Vec<Type>),

    /// Intersection type: T1 & T2 & ... & Tn
    Intersection(Vec<Type>),

    /// Type alias (resolved to target type)
    Alias {
        name: InternedString,
        target: Box<Type>,
    },

    /// Associated type (for traits/interfaces)
    Associated {
        trait_name: InternedString,
        type_name: InternedString,
    },

    /// Higher-kinded type: F<_> where F is a type constructor
    HigherKinded {
        constructor: InternedString,
        arity: usize,
        kind: Kind,
    },

    /// Projection type: T::AssocType
    Projection {
        base: Box<Type>,
        item: InternedString,
    },

    /// Index type: T[K]
    Index { base: Box<Type>, index: Box<Type> },

    /// The top type (any/unknown)
    Any,

    /// Error type (for error recovery)
    Error,

    /// Self type (for trait method definitions)
    SelfType,

    /// Explicit nullable type: T? or Option<T>
    Nullable(Box<Type>),

    /// Explicit non-null type
    NonNull(Box<Type>),

    /// Const variable for const generics
    ConstVar {
        id: ConstVarId,
        name: Option<InternedString>,
        const_type: Box<Type>,
    },

    /// Const-dependent type with constraints
    ConstDependent {
        base_type: Box<Type>,
        constraint: ConstConstraint,
    },

    /// Dynamic type for gradual typing
    Dynamic,

    /// Unknown type for gradual typing (TypeScript unknown)
    Unknown,

    /// Structural interface type (Go/TypeScript style)
    Interface {
        methods: Vec<MethodSig>,
        is_structural: bool, // true for Go-style, false for nominal
        nullability: NullabilityKind,
    },

    /// Anonymous struct type (structural typing)
    Struct {
        fields: Vec<FieldDef>,
        is_anonymous: bool,
        nullability: NullabilityKind,
    },

    /// Trait type (Rust-style)
    Trait {
        id: TypeId,
        associated_types: Vec<(InternedString, Type)>,
        super_traits: Vec<Type>,
    },

    /// External/Opaque type (for FFI, ZRTL plugins)
    /// At the HIR level, this becomes a pointer to an opaque type
    Extern {
        name: InternedString,
        layout: Option<ExternLayout>,
    },

    /// Unresolved type name (placeholder from parser, to be resolved by compiler)
    /// The compiler's type resolution pass will look this up in TypeRegistry aliases
    /// and replace it with the actual type (e.g., Type::Extern)
    Unresolved(InternedString),
}

/// Layout information for extern types (optional)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternLayout {
    pub size: usize,
    pub align: usize,
}

/// Type variable for inference
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar {
    pub id: TypeVarId,
    pub name: Option<InternedString>,
    pub kind: TypeVarKind,
}

/// Unique identifier for type variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVarId(u32);

impl TypeVarId {
    pub fn next() -> Self {
        static NEXT_TYPE_VAR: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_TYPE_VAR.fetch_add(1, Ordering::SeqCst))
    }

    pub fn unknown() -> Self {
        TypeVarId(0)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl TypeVar {
    /// Create a new unbound type variable with the given name
    pub fn unbound(name: InternedString) -> Self {
        Self {
            id: TypeVarId::next(),
            name: Some(name),
            kind: TypeVarKind::Type,
        }
    }

    /// Create a new anonymous unbound type variable
    pub fn fresh() -> Self {
        Self {
            id: TypeVarId::next(),
            name: None,
            kind: TypeVarKind::Type,
        }
    }
}

/// Kind of type variable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeVarKind {
    /// Regular type variable
    Type,
    /// Lifetime variable
    Lifetime,
    /// Const variable (for const generics)
    Const,
    /// Integer type variable (for numeric inference)
    Integral,
    /// Float type variable (for float inference)
    Floating,
    /// Any numeric type variable
    Numeric,
}

/// Definition of a registered type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeDefinition {
    pub id: TypeId,
    pub name: InternedString,
    pub kind: TypeKind,
    pub type_params: Vec<TypeParam>,
    pub constraints: Vec<TypeConstraint>,
    pub fields: Vec<FieldDef>,
    pub methods: Vec<MethodSig>,
    pub constructors: Vec<ConstructorSig>,
    pub metadata: TypeMetadata,
    pub span: Span,
}

/// Kind of registered type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeKind {
    /// Primitive/atomic type (user-defined, e.g., "int", "str", "bool")
    Atomic,
    /// Struct/Record type
    Struct {
        fields: Vec<FieldDef>,
        is_tuple: bool,
    },
    /// Class type
    Class,
    /// Interface/Trait type
    Interface {
        methods: Vec<MethodSig>,
        associated_types: Vec<AssociatedTypeDef>,
        super_traits: Vec<Type>,
    },
    /// Enum/Sum type
    Enum { variants: Vec<VariantDef> },
    /// Type alias
    Alias { target: Type },
    /// Abstract type (like Haxe's abstract) - wrapper with implicit conversions
    /// Provides zero-cost abstraction with compile-time conversions to/from underlying type
    Abstract {
        /// The underlying/base type this abstracts over
        underlying_type: Type,
        /// Types this abstract can implicitly convert TO (via to() methods or From impls)
        implicit_to: Vec<Type>,
        /// Types this abstract can implicitly convert FROM (via from() methods or Into impls)
        implicit_from: Vec<Type>,
        /// Suffixes for literal syntax (e.g., ["ms", "s", "ns"] for Duration allows 1ms, 2s, 3ns syntax)
        /// Each suffix maps to specific from_* method (e.g., Suffix("ms") -> from_ms, Suffix("s") -> from_s)
        suffixes: Vec<String>,
    },
    /// Function type constructor
    Function,
    /// Array/Collection type constructor
    Array,
    /// Generic type constructor
    Generic,
}

/// Type parameter definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeParam {
    pub name: InternedString,
    pub bounds: Vec<TypeBound>,
    pub variance: Variance,
    pub default: Option<Type>,
    pub span: Span,
}

/// Type constraints that can be applied to types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Subtyping constraint: A <: B
    Subtype { sub: Type, super_type: Type },
    /// Equality constraint: A = B
    Equality { left: Type, right: Type },
    /// Trait/Interface implementation: T : Trait
    Implementation { ty: Type, trait_id: TypeId },
    /// Has member constraint: T has field/method
    HasMember {
        ty: Type,
        member: InternedString,
        member_type: Type,
    },
    /// Callable constraint: T is callable with signature
    Callable { ty: Type, signature: Type },
    /// Custom constraint (language-specific)
    Custom {
        name: InternedString,
        args: Vec<Type>,
    },
}

/// Field definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: InternedString,
    pub ty: Type,
    pub visibility: Visibility,
    pub mutability: Mutability,
    pub is_static: bool,
    pub span: Span,
    /// Property getter (for C#/Kotlin style properties)
    pub getter: Option<Box<MethodSig>>,
    /// Property setter (for C#/Kotlin style properties)
    pub setter: Option<Box<MethodSig>>,
    /// Whether this is a compiler-generated property
    pub is_synthetic: bool,
}

/// Method signature
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MethodSig {
    pub name: InternedString,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<ParamDef>,
    pub return_type: Type,
    pub where_clause: Vec<TypeConstraint>,
    pub is_static: bool,
    pub is_async: bool,
    pub visibility: Visibility,
    pub span: Span,
    /// Whether this is an extension method (C#/Kotlin/Swift style)
    pub is_extension: bool,
}

/// Constructor signature
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstructorSig {
    pub type_params: Vec<TypeParam>,
    pub params: Vec<ParamDef>,
    pub visibility: Visibility,
    pub span: Span,
}

/// Parameter definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: InternedString,
    pub ty: Type,
    pub is_self: bool,
    pub is_varargs: bool,
    pub is_mut: bool,
}

/// Parameter information for function types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParamInfo {
    pub name: Option<InternedString>,
    pub ty: Type,
    pub is_optional: bool,
    pub is_varargs: bool,
    pub is_keyword_only: bool,
    pub is_positional_only: bool,
    pub is_out: bool,
    pub is_ref: bool,
    pub is_inout: bool,
}

/// Associated type definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssociatedTypeDef {
    pub name: InternedString,
    pub bounds: Vec<TypeBound>,
    pub default: Option<Type>,
}

/// Enum variant definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantDef {
    pub name: InternedString,
    pub fields: VariantFields,
    pub discriminant: Option<i64>,
    pub span: Span,
}

/// Variant field types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariantFields {
    /// Unit variant: `None`
    Unit,
    /// Tuple variant: `Some(i32, String)`
    Tuple(Vec<Type>),
    /// Named variant: `Point { x: i32, y: i32 }`
    Named(Vec<FieldDef>),
}

/// Trait definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitDef {
    pub id: TypeId,
    pub name: InternedString,
    pub type_params: Vec<TypeParam>,
    pub super_traits: Vec<Type>,
    pub methods: Vec<MethodSig>,
    pub associated_types: Vec<AssociatedTypeDef>,
    pub is_object_safe: bool,
    pub span: Span,
}

/// Trait implementation definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplDef {
    pub trait_id: TypeId,
    pub for_type: Type,
    pub type_args: Vec<Type>,
    pub methods: Vec<MethodImpl>,
    pub associated_types: HashMap<InternedString, Type>,
    pub where_clause: Vec<TypeConstraint>,
    pub span: Span,
}

/// Method implementation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodImpl {
    pub signature: MethodSig,
    pub is_default: bool,
}

/// Lifetime bounds
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifetimeBound {
    /// 'a: 'b (a outlives b)
    Outlives(Lifetime),
    /// 'static
    Static,
}

/// Type bounds/constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeBound {
    /// Trait bound: T: Trait
    Trait {
        name: InternedString,
        args: Vec<Type>,
    },
    /// Lifetime bound: T: 'a
    Lifetime(Lifetime),
    /// Equality bound: T::Assoc = Type
    Equality(Type),
    /// Subtype bound: T <: Super
    Subtype(Type),
    /// Supertype bound: T :> Sub
    Supertype(Type),
    /// Constructor bound: T: new(...)
    Constructor(Vec<Type>),
    /// Sized bound: T: Sized
    Sized,
    /// Copy bound: T: Copy
    Copy,
    /// Send bound: T: Send
    Send,
    /// Sync bound: T: Sync
    Sync,
    /// Static lifetime bound: T: 'static
    Static,
    /// Reference type bound: T: ref
    ReferenceType,
    /// Value type bound: T: val
    ValueType,
    /// Unmanaged bound: T: unmanaged
    Unmanaged,
    /// Higher-ranked trait bound: for<'a> T: Trait<'a>
    HigherRanked {
        lifetimes: Vec<InternedString>,
        bound: Box<TypeBound>,
    },
    /// Custom bound with parameters
    Custom {
        name: InternedString,
        args: Vec<Type>,
    },
}

/// Variance for type parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Variance {
    /// Covariant: can substitute subtype for supertype
    Covariant,
    /// Contravariant: can substitute supertype for subtype
    Contravariant,
    /// Invariant: no substitution allowed
    Invariant,
    /// Bivariant: both covariant and contravariant
    Bivariant,
}

/// Mutability annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Mutability {
    #[default]
    Immutable,
    Mutable,
}

/// Visibility annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Visibility {
    Public,
    #[default]
    Private,
    Protected,
    Internal,
}

/// Lifetime annotation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Lifetime {
    pub name: InternedString,
    pub bounds: Vec<LifetimeBound>,
}

impl Lifetime {
    pub fn named(name: InternedString) -> Self {
        Self {
            name,
            bounds: Vec::new(),
        }
    }

    pub fn with_bounds(name: InternedString, bounds: Vec<LifetimeBound>) -> Self {
        Self { name, bounds }
    }
}

/// Metadata associated with types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeMetadata {
    /// Whether the type can be instantiated directly
    pub is_instantiable: bool,
    /// Whether the type supports inheritance
    pub is_inheritable: bool,
    /// Whether the type is abstract
    pub is_abstract: bool,
    /// Whether the type is final (cannot be inherited)
    pub is_final: bool,
    /// Whether the type supports null values
    pub is_nullable: bool,
    /// Size of the type in bytes (if known)
    pub size_bytes: Option<usize>,
    /// Alignment requirements
    pub alignment: Option<usize>,
    /// Custom metadata (language-specific)
    pub custom: HashMap<InternedString, String>,
}

impl Default for TypeMetadata {
    fn default() -> Self {
        Self {
            is_instantiable: true,
            is_inheritable: false,
            is_abstract: false,
            is_final: false,
            is_nullable: false,
            size_bytes: None,
            alignment: None,
            custom: HashMap::new(),
        }
    }
}

/// The type registry that manages all registered types
#[derive(Debug, Clone)]
pub struct TypeRegistry {
    types: HashMap<TypeId, TypeDefinition>,
    name_to_id: HashMap<InternedString, TypeId>,
    aliases: HashMap<InternedString, Type>,

    // Trait system
    traits: HashMap<TypeId, TraitDef>,
    trait_name_to_id: HashMap<InternedString, TypeId>,
    implementations: HashMap<TypeId, Vec<ImplDef>>, // trait_id -> implementations
    impl_cache: HashMap<(TypeId, Type), ImplDef>,   // (trait_id, for_type) -> impl for fast lookup
    trait_hierarchy: HashMap<TypeId, Vec<TypeId>>,  // trait -> super traits

    // Coherence and constraints
    coherence_graph: HashMap<(TypeId, TypeId), Vec<TypeConstraint>>, // (type, trait) -> constraints
    type_implementations: HashMap<TypeId, HashSet<TypeId>>,          // type -> implemented traits
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            name_to_id: HashMap::new(),
            aliases: HashMap::new(),
            traits: HashMap::new(),
            trait_name_to_id: HashMap::new(),
            implementations: HashMap::new(),
            impl_cache: HashMap::new(),
            trait_hierarchy: HashMap::new(),
            coherence_graph: HashMap::new(),
            type_implementations: HashMap::new(),
        }
    }

    /// Register a new type definition
    pub fn register_type(&mut self, type_def: TypeDefinition) -> TypeId {
        let id = type_def.id;
        self.name_to_id.insert(type_def.name, id);
        self.types.insert(id, type_def);
        id
    }

    /// Create and register a new atomic type (like int, bool, string)
    pub fn register_atomic_type(
        &mut self,
        name: InternedString,
        metadata: TypeMetadata,
        span: Span,
    ) -> TypeId {
        let id = TypeId::next();
        let type_def = TypeDefinition {
            id,
            name,
            kind: TypeKind::Atomic,
            type_params: vec![],
            constraints: vec![],
            fields: vec![],
            methods: vec![],
            constructors: vec![],
            metadata,
            span,
        };
        self.register_type(type_def)
    }

    /// Create and register a new struct type
    pub fn register_struct_type(
        &mut self,
        name: InternedString,
        type_params: Vec<TypeParam>,
        fields: Vec<FieldDef>,
        methods: Vec<MethodSig>,
        constructors: Vec<ConstructorSig>,
        metadata: TypeMetadata,
        span: Span,
    ) -> TypeId {
        let id = TypeId::next();
        let type_def = TypeDefinition {
            id,
            name,
            kind: TypeKind::Struct {
                fields: fields.clone(),
                is_tuple: false,
            },
            type_params,
            constraints: vec![],
            fields,
            methods,
            constructors,
            metadata,
            span,
        };
        self.register_type(type_def)
    }

    /// Create and register a new enum type
    pub fn register_enum_type(
        &mut self,
        name: InternedString,
        type_params: Vec<TypeParam>,
        variants: Vec<VariantDef>,
        methods: Vec<MethodSig>,
        metadata: TypeMetadata,
        span: Span,
    ) -> TypeId {
        let id = TypeId::next();
        let type_def = TypeDefinition {
            id,
            name,
            kind: TypeKind::Enum {
                variants: variants.clone(),
            },
            type_params,
            constraints: vec![],
            fields: vec![],
            methods,
            constructors: vec![],
            metadata,
            span,
        };
        self.register_type(type_def)
    }

    /// Register a type alias
    pub fn register_alias(&mut self, name: InternedString, target: Type) {
        self.aliases.insert(name, target);
    }

    /// Register a trait definition
    pub fn register_trait(&mut self, trait_def: TraitDef) -> TypeId {
        let id = trait_def.id;
        self.trait_name_to_id.insert(trait_def.name, id);

        // Build trait hierarchy
        let super_trait_ids: Vec<TypeId> = trait_def
            .super_traits
            .iter()
            .filter_map(|super_trait| {
                if let Type::Named { id, .. } = super_trait {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        self.trait_hierarchy.insert(id, super_trait_ids);

        self.traits.insert(id, trait_def);
        id
    }

    /// Register a trait implementation
    pub fn register_implementation(&mut self, impl_def: ImplDef) {
        let trait_id = impl_def.trait_id;
        let for_type = impl_def.for_type.clone();

        // Add to implementations list
        self.implementations
            .entry(trait_id)
            .or_insert_with(Vec::new)
            .push(impl_def.clone());

        // Add to cache for fast lookup
        self.impl_cache
            .insert((trait_id, for_type.clone()), impl_def);

        // Update type implementations tracking
        if let Type::Named { id: type_id, .. } = &for_type {
            self.type_implementations
                .entry(*type_id)
                .or_insert_with(HashSet::new)
                .insert(trait_id);
        }
    }

    /// Merge types, traits, and implementations from another TypeRegistry
    pub fn merge_from(&mut self, other: &TypeRegistry) {
        // Merge type definitions
        for (type_id, type_def) in &other.types {
            self.types.insert(*type_id, type_def.clone());
            self.name_to_id.insert(type_def.name, *type_id);
        }

        // Merge type aliases
        for (name, alias_type) in &other.aliases {
            self.aliases.insert(*name, alias_type.clone());
        }

        // Merge traits
        for (trait_id, trait_def) in &other.traits {
            self.traits.insert(*trait_id, trait_def.clone());
            self.trait_name_to_id.insert(trait_def.name, *trait_id);
        }

        // Merge trait implementations
        for (trait_id, impls) in &other.implementations {
            let impl_list = self
                .implementations
                .entry(*trait_id)
                .or_insert_with(Vec::new);
            for impl_def in impls {
                impl_list.push(impl_def.clone());
                self.impl_cache
                    .insert((*trait_id, impl_def.for_type.clone()), impl_def.clone());

                // Update type implementations tracking
                if let Type::Named { id: type_id, .. } = &impl_def.for_type {
                    self.type_implementations
                        .entry(*type_id)
                        .or_insert_with(HashSet::new)
                        .insert(*trait_id);
                }
            }
        }

        // Merge trait hierarchy
        for (trait_id, super_traits) in &other.trait_hierarchy {
            let hierarchy = self
                .trait_hierarchy
                .entry(*trait_id)
                .or_insert_with(Vec::new);
            for super_trait in super_traits {
                if !hierarchy.contains(super_trait) {
                    hierarchy.push(*super_trait);
                }
            }
        }

        // Merge coherence graph
        for ((type_id, trait_id), constraints) in &other.coherence_graph {
            let constraint_list = self
                .coherence_graph
                .entry((*type_id, *trait_id))
                .or_insert_with(Vec::new);
            for constraint in constraints {
                constraint_list.push(constraint.clone());
            }
        }
    }

    /// Look up a type by name
    pub fn get_type_by_name(&self, name: InternedString) -> Option<&TypeDefinition> {
        self.name_to_id.get(&name).and_then(|id| self.types.get(id))
    }

    /// Look up a type by ID
    pub fn get_type_by_id(&self, id: TypeId) -> Option<&TypeDefinition> {
        self.types.get(&id)
    }

    /// Resolve a type alias
    pub fn resolve_alias(&self, name: InternedString) -> Option<&Type> {
        self.aliases.get(&name)
    }

    /// Look up a trait by name
    pub fn get_trait_by_name(&self, name: InternedString) -> Option<&TraitDef> {
        self.trait_name_to_id
            .get(&name)
            .and_then(|id| self.traits.get(id))
    }

    /// Look up a trait by ID
    pub fn get_trait_by_id(&self, id: TypeId) -> Option<&TraitDef> {
        self.traits.get(&id)
    }

    /// Alias for get_trait_by_id - used by trait dispatch lowering
    pub fn get_trait_def(&self, id: TypeId) -> Option<&TraitDef> {
        self.get_trait_by_id(id)
    }

    /// Check if a type implements a trait/interface (including super-traits)
    pub fn type_implements(&self, for_type: &Type, trait_id: TypeId) -> bool {
        // Direct implementation check
        if self.impl_cache.contains_key(&(trait_id, for_type.clone())) {
            return true;
        }

        // Check super-traits recursively
        if let Some(super_traits) = self.trait_hierarchy.get(&trait_id) {
            for super_trait_id in super_traits {
                if self.type_implements(for_type, *super_trait_id) {
                    return true;
                }
            }
        }

        false
    }

    /// Get all traits/interfaces implemented by a type
    pub fn get_implementations(&self, for_type: &Type) -> Vec<TypeId> {
        let mut implemented_traits = HashSet::new();

        // Find direct implementations
        for (trait_id, impls) in &self.implementations {
            for impl_def in impls {
                if impl_def.for_type == *for_type {
                    implemented_traits.insert(*trait_id);

                    // Add super-traits recursively
                    self.collect_super_traits(*trait_id, &mut implemented_traits);
                }
            }
        }

        implemented_traits.into_iter().collect()
    }

    /// Get super-traits of a trait recursively
    pub fn get_super_traits(&self, trait_id: TypeId) -> Vec<TypeId> {
        let mut super_traits = HashSet::new();
        self.collect_super_traits(trait_id, &mut super_traits);
        super_traits.into_iter().collect()
    }

    /// Helper method to recursively collect super-traits
    fn collect_super_traits(&self, trait_id: TypeId, collected: &mut HashSet<TypeId>) {
        if let Some(super_traits) = self.trait_hierarchy.get(&trait_id) {
            for super_trait_id in super_traits {
                if collected.insert(*super_trait_id) {
                    self.collect_super_traits(*super_trait_id, collected);
                }
            }
        }
    }

    /// Get all registered types
    pub fn get_all_types(&self) -> impl Iterator<Item = &TypeDefinition> {
        self.types.values()
    }

    /// Get the implementation of a trait for a specific type
    pub fn get_implementation(&self, for_type: &Type, trait_id: TypeId) -> Option<&ImplDef> {
        self.impl_cache.get(&(trait_id, for_type.clone()))
    }

    /// Add a method to an existing type definition (for inherent methods)
    pub fn add_method_to_type(&mut self, type_id: TypeId, method: MethodSig) {
        if let Some(type_def) = self.types.get_mut(&type_id) {
            type_def.methods.push(method);
        }
    }

    /// Iterate over all trait implementations
    /// Returns an iterator over (trait_id, implementations) pairs
    pub fn iter_implementations(&self) -> impl Iterator<Item = (&TypeId, &Vec<ImplDef>)> {
        self.implementations.iter()
    }

    /// Create a Type::Named instance for a given type
    pub fn make_type(&self, type_id: TypeId, type_args: Vec<Type>) -> Type {
        Type::Named {
            id: type_id,
            type_args,
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::default(),
        }
    }

    /// Create a Type::Named instance for a given type name
    pub fn make_type_by_name(&self, name: InternedString, type_args: Vec<Type>) -> Option<Type> {
        self.name_to_id.get(&name).map(|&id| Type::Named {
            id,
            type_args,
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::default(),
        })
    }

    /// Create a Type::Named instance with full universal features
    pub fn make_universal_type(
        &self,
        type_id: TypeId,
        type_args: Vec<Type>,
        const_args: Vec<ConstValue>,
        variance: Vec<Variance>,
        nullability: NullabilityKind,
    ) -> Type {
        Type::Named {
            id: type_id,
            type_args,
            const_args,
            variance,
            nullability,
        }
    }

    /// Register the built-in Result<T, E> enum type
    ///
    /// Result is a generic enum with two variants:
    /// - Ok(T): Success case containing a value of type T
    /// - Err(E): Error case containing an error of type E
    pub fn register_result_type(
        &mut self,
        arena: &mut crate::arena::AstArena,
        span: Span,
    ) -> TypeId {
        let result_name = arena.intern_string("Result");
        let t_param_name = arena.intern_string("T");
        let e_param_name = arena.intern_string("E");
        let ok_variant_name = arena.intern_string("Ok");
        let err_variant_name = arena.intern_string("Err");

        // Define type parameters: T and E
        let type_params = vec![
            TypeParam {
                name: t_param_name,
                bounds: vec![],
                variance: Variance::Covariant,
                default: None,
                span,
            },
            TypeParam {
                name: e_param_name,
                bounds: vec![],
                variance: Variance::Covariant,
                default: None,
                span,
            },
        ];

        // Define variants: Ok(T) and Err(E)
        let variants = vec![
            VariantDef {
                name: ok_variant_name,
                fields: VariantFields::Tuple(vec![Type::TypeVar(TypeVar::unbound(t_param_name))]),
                discriminant: Some(0),
                span,
            },
            VariantDef {
                name: err_variant_name,
                fields: VariantFields::Tuple(vec![Type::TypeVar(TypeVar::unbound(e_param_name))]),
                discriminant: Some(1),
                span,
            },
        ];

        self.register_enum_type(
            result_name,
            type_params,
            variants,
            vec![], // No methods for now
            TypeMetadata::default(),
            span,
        )
    }

    /// Register the built-in Option<T> enum type
    ///
    /// Option is a generic enum with two variants:
    /// - Some(T): Contains a value of type T
    /// - None: No value present
    pub fn register_option_type(
        &mut self,
        arena: &mut crate::arena::AstArena,
        span: Span,
    ) -> TypeId {
        let option_name = arena.intern_string("Option");
        let t_param_name = arena.intern_string("T");
        let some_variant_name = arena.intern_string("Some");
        let none_variant_name = arena.intern_string("None");

        // Define type parameter: T
        let type_params = vec![TypeParam {
            name: t_param_name,
            bounds: vec![],
            variance: Variance::Covariant,
            default: None,
            span,
        }];

        // Define variants: Some(T) and None
        let variants = vec![
            VariantDef {
                name: some_variant_name,
                fields: VariantFields::Tuple(vec![Type::TypeVar(TypeVar::unbound(t_param_name))]),
                discriminant: Some(0),
                span,
            },
            VariantDef {
                name: none_variant_name,
                fields: VariantFields::Unit,
                discriminant: Some(1),
                span,
            },
        ];

        self.register_enum_type(
            option_name,
            type_params,
            variants,
            vec![], // No methods for now
            TypeMetadata::default(),
            span,
        )
    }
}

impl Type {
    /// Check if this type is nullable
    pub fn is_nullable(&self) -> bool {
        match self {
            Type::Nullable(_) => true,
            Type::Named { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Function { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Array { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Reference { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Interface { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Struct { nullability, .. } => *nullability == NullabilityKind::Nullable,
            Type::Any | Type::Dynamic => true,
            _ => false,
        }
    }

    /// Make this type nullable
    pub fn make_nullable(self) -> Self {
        if self.is_nullable() {
            self
        } else {
            Type::Nullable(Box::new(self))
        }
    }

    /// Make this type non-null
    pub fn make_non_null(self) -> Self {
        match self {
            Type::Nullable(inner) => *inner,
            _ => Type::NonNull(Box::new(self)),
        }
    }

    /// Get the underlying type, stripping nullability wrappers
    pub fn underlying_type(&self) -> &Self {
        match self {
            Type::Nullable(inner) | Type::NonNull(inner) => inner.underlying_type(),
            _ => self,
        }
    }

    /// Check if this type supports const generics
    pub fn supports_const_generics(&self) -> bool {
        match self {
            Type::Named { const_args, .. } => !const_args.is_empty(),
            Type::Array { size, .. } => size.is_some(),
            Type::ConstVar { .. } | Type::ConstDependent { .. } => true,
            _ => false,
        }
    }

    /// Check if this type is async
    pub fn is_async(&self) -> bool {
        match self {
            Type::Function { async_kind, .. } => *async_kind != AsyncKind::Sync,
            _ => false,
        }
    }

    /// Check if this is a structural type
    pub fn is_structural(&self) -> bool {
        match self {
            Type::Interface { is_structural, .. } => *is_structural,
            Type::Struct { is_anonymous, .. } => *is_anonymous,
            _ => false,
        }
    }

    /// Check if this is a gradual typing type (dynamic/unknown)
    pub fn is_gradual(&self) -> bool {
        matches!(self, Type::Any | Type::Dynamic | Type::Unknown)
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Well-known operator trait IDs for compiler dispatch
#[cfg(feature = "operator_traits")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinTraitIds {
    pub add: TypeId,
    pub sub: TypeId,
    pub mul: TypeId,
    pub div: TypeId,
    pub modulo: TypeId,
    pub neg: TypeId,
    pub matmul: TypeId,
    pub eq: TypeId,
    pub ord: TypeId,
    pub bit_and: TypeId,
    pub bit_or: TypeId,
    pub bit_xor: TypeId,
    pub not: TypeId,
    pub index: TypeId,
    pub index_mut: TypeId,
    pub display: TypeId,
    pub debug: TypeId,
    pub clone: TypeId,
    pub drop: TypeId,
    pub iterator: TypeId,
    pub into_iterator: TypeId,
}

#[cfg(feature = "operator_traits")]
impl TypeRegistry {
    /// Register all built-in operator traits and return their IDs
    ///
    /// This creates the standard operator traits that the compiler uses
    /// for operator overloading. When seeing `a + b`, the compiler looks
    /// for `Add` trait impl on the type of `a`.
    pub fn register_builtin_traits(
        &mut self,
        arena: &mut crate::arena::AstArena,
    ) -> BuiltinTraitIds {
        let span = Span::default();

        // Create common type names
        let self_name = arena.intern_string("Self");
        let rhs_name = arena.intern_string("Rhs");
        let output_name = arena.intern_string("Output");
        let item_name = arena.intern_string("Item");

        // Self type reference
        let self_type = Type::TypeVar(TypeVar::unbound(self_name));

        // --- Arithmetic Operators ---

        // Add<Rhs> trait - enables a + b
        let add_id =
            self.register_operator_trait(arena, "Add", rhs_name, output_name, "add", &span);

        // Sub<Rhs> trait - enables a - b
        let sub_id =
            self.register_operator_trait(arena, "Sub", rhs_name, output_name, "sub", &span);

        // Mul<Rhs> trait - enables a * b
        let mul_id =
            self.register_operator_trait(arena, "Mul", rhs_name, output_name, "mul", &span);

        // Div<Rhs> trait - enables a / b
        let div_id =
            self.register_operator_trait(arena, "Div", rhs_name, output_name, "div", &span);

        // Mod<Rhs> trait - enables a % b
        let modulo_id =
            self.register_operator_trait(arena, "Mod", rhs_name, output_name, "mod", &span);

        // MatMul<Rhs> trait - enables a @ b (matrix multiplication)
        let matmul_id =
            self.register_operator_trait(arena, "MatMul", rhs_name, output_name, "matmul", &span);

        // Neg trait - enables -a (unary negation)
        let neg_id = self.register_unary_operator_trait(arena, "Neg", output_name, "neg", &span);

        // --- Comparison Operators ---

        // Eq<Rhs> trait - enables a == b, a != b
        let eq_id = {
            let trait_name = arena.intern_string("Eq");
            let eq_method = arena.intern_string("eq");
            let ne_method = arena.intern_string("ne");
            let id = TypeId::next();

            let rhs_type = Type::TypeVar(TypeVar::unbound(rhs_name));
            let bool_type = Type::Primitive(PrimitiveType::Bool);

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![TypeParam {
                    name: rhs_name,
                    bounds: vec![],
                    variance: Variance::Invariant,
                    default: Some(self_type.clone()),
                    span,
                }],
                super_traits: vec![],
                methods: vec![
                    MethodSig {
                        name: eq_method,
                        type_params: vec![],
                        params: vec![
                            ParamDef {
                                name: self_name,
                                ty: self_type.clone(),
                                is_self: true,
                                is_varargs: false,
                                is_mut: false,
                            },
                            ParamDef {
                                name: rhs_name,
                                ty: rhs_type.clone(),
                                is_self: false,
                                is_varargs: false,
                                is_mut: false,
                            },
                        ],
                        return_type: bool_type.clone(),
                        where_clause: vec![],
                        is_static: false,
                        is_async: false,
                        visibility: Visibility::Public,
                        span,
                        is_extension: false,
                    },
                    MethodSig {
                        name: ne_method,
                        type_params: vec![],
                        params: vec![
                            ParamDef {
                                name: self_name,
                                ty: self_type.clone(),
                                is_self: true,
                                is_varargs: false,
                                is_mut: false,
                            },
                            ParamDef {
                                name: rhs_name,
                                ty: rhs_type,
                                is_self: false,
                                is_varargs: false,
                                is_mut: false,
                            },
                        ],
                        return_type: bool_type,
                        where_clause: vec![],
                        is_static: false,
                        is_async: false,
                        visibility: Visibility::Public,
                        span,
                        is_extension: false,
                    },
                ],
                associated_types: vec![],
                is_object_safe: true,
                span,
            };
            self.register_trait(trait_def)
        };

        // Ord<Rhs> trait - enables a < b, a > b, a <= b, a >= b
        let ord_id = {
            let trait_name = arena.intern_string("Ord");
            let lt_method = arena.intern_string("lt");
            let gt_method = arena.intern_string("gt");
            let le_method = arena.intern_string("le");
            let ge_method = arena.intern_string("ge");
            let id = TypeId::next();

            let rhs_type = Type::TypeVar(TypeVar::unbound(rhs_name));
            let bool_type = Type::Primitive(PrimitiveType::Bool);

            let comparison_method = |name: InternedString| MethodSig {
                name,
                type_params: vec![],
                params: vec![
                    ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    },
                    ParamDef {
                        name: rhs_name,
                        ty: rhs_type.clone(),
                        is_self: false,
                        is_varargs: false,
                        is_mut: false,
                    },
                ],
                return_type: bool_type.clone(),
                where_clause: vec![],
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span,
                is_extension: false,
            };

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![TypeParam {
                    name: rhs_name,
                    bounds: vec![],
                    variance: Variance::Invariant,
                    default: Some(self_type.clone()),
                    span,
                }],
                super_traits: vec![],
                methods: vec![
                    comparison_method(lt_method),
                    comparison_method(gt_method),
                    comparison_method(le_method),
                    comparison_method(ge_method),
                ],
                associated_types: vec![],
                is_object_safe: true,
                span,
            };
            self.register_trait(trait_def)
        };

        // --- Bitwise Operators ---

        // BitAnd<Rhs> trait - enables a & b
        let bit_and_id =
            self.register_operator_trait(arena, "BitAnd", rhs_name, output_name, "bitand", &span);

        // BitOr<Rhs> trait - enables a | b
        let bit_or_id =
            self.register_operator_trait(arena, "BitOr", rhs_name, output_name, "bitor", &span);

        // BitXor<Rhs> trait - enables a ^ b
        let bit_xor_id =
            self.register_operator_trait(arena, "BitXor", rhs_name, output_name, "bitxor", &span);

        // Not trait - enables !a (logical/bitwise not)
        let not_id = self.register_unary_operator_trait(arena, "Not", output_name, "not", &span);

        // --- Indexing Operators ---

        // Index<Idx> trait - enables a[i] (read)
        let index_id = {
            let trait_name = arena.intern_string("Index");
            let idx_name = arena.intern_string("Idx");
            let index_method = arena.intern_string("index");
            let id = TypeId::next();

            let idx_type = Type::TypeVar(TypeVar::unbound(idx_name));
            let output_type = Type::TypeVar(TypeVar::unbound(output_name));

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![TypeParam {
                    name: idx_name,
                    bounds: vec![],
                    variance: Variance::Contravariant,
                    default: None,
                    span,
                }],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: index_method,
                    type_params: vec![],
                    params: vec![
                        ParamDef {
                            name: self_name,
                            ty: self_type.clone(),
                            is_self: true,
                            is_varargs: false,
                            is_mut: false,
                        },
                        ParamDef {
                            name: idx_name,
                            ty: idx_type,
                            is_self: false,
                            is_varargs: false,
                            is_mut: false,
                        },
                    ],
                    return_type: output_type,
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![AssociatedTypeDef {
                    name: output_name,
                    bounds: vec![],
                    default: None,
                }],
                is_object_safe: false, // Associated type makes it non-object-safe
                span,
            };
            self.register_trait(trait_def)
        };

        // IndexMut<Idx> trait - enables a[i] = v (write)
        let index_mut_id = {
            let trait_name = arena.intern_string("IndexMut");
            let idx_name = arena.intern_string("Idx");
            let index_mut_method = arena.intern_string("index_mut");
            let id = TypeId::next();

            let idx_type = Type::TypeVar(TypeVar::unbound(idx_name));
            let output_type = Type::TypeVar(TypeVar::unbound(output_name));

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![TypeParam {
                    name: idx_name,
                    bounds: vec![],
                    variance: Variance::Contravariant,
                    default: None,
                    span,
                }],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: index_mut_method,
                    type_params: vec![],
                    params: vec![
                        ParamDef {
                            name: self_name,
                            ty: self_type.clone(),
                            is_self: true,
                            is_varargs: false,
                            is_mut: true,
                        },
                        ParamDef {
                            name: idx_name,
                            ty: idx_type,
                            is_self: false,
                            is_varargs: false,
                            is_mut: false,
                        },
                    ],
                    return_type: Type::Reference {
                        ty: Box::new(output_type),
                        mutability: Mutability::Mutable,
                        lifetime: None,
                        nullability: NullabilityKind::NonNull,
                    },
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![AssociatedTypeDef {
                    name: output_name,
                    bounds: vec![],
                    default: None,
                }],
                is_object_safe: false,
                span,
            };
            self.register_trait(trait_def)
        };

        // --- Formatting Traits ---

        // Display trait - enables to_string()
        let display_id = {
            let trait_name = arena.intern_string("Display");
            let to_string_method = arena.intern_string("to_string");
            let id = TypeId::next();

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: to_string_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    }],
                    return_type: Type::Primitive(PrimitiveType::String),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![],
                is_object_safe: true,
                span,
            };
            self.register_trait(trait_def)
        };

        // Debug trait - enables debug_string()
        let debug_id = {
            let trait_name = arena.intern_string("Debug");
            let debug_string_method = arena.intern_string("debug_string");
            let id = TypeId::next();

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: debug_string_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    }],
                    return_type: Type::Primitive(PrimitiveType::String),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![],
                is_object_safe: true,
                span,
            };
            self.register_trait(trait_def)
        };

        // --- Lifecycle Traits ---

        // Clone trait - enables clone()
        let clone_id = {
            let trait_name = arena.intern_string("Clone");
            let clone_method = arena.intern_string("clone");
            let id = TypeId::next();

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: clone_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    }],
                    return_type: self_type.clone(),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![],
                is_object_safe: false, // Returns Self
                span,
            };
            self.register_trait(trait_def)
        };

        // Drop trait - enables custom cleanup
        let drop_id = {
            let trait_name = arena.intern_string("Drop");
            let drop_method = arena.intern_string("drop");
            let id = TypeId::next();

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: drop_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: true,
                    }],
                    return_type: Type::Primitive(PrimitiveType::Unit),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![],
                is_object_safe: true,
                span,
            };
            self.register_trait(trait_def)
        };

        // --- Iterator Traits ---

        // Iterator trait - enables next()
        let iterator_id = {
            let trait_name = arena.intern_string("Iterator");
            let next_method = arena.intern_string("next");
            let id = TypeId::next();

            let item_type = Type::TypeVar(TypeVar::unbound(item_name));

            // Option<Item> for return type
            let option_name = arena.intern_string("Option");
            let option_type = Type::Named {
                id: TypeId::next(), // This should be resolved to actual Option type
                type_args: vec![item_type.clone()],
                const_args: vec![],
                variance: vec![],
                nullability: NullabilityKind::default(),
            };

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: next_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type.clone(),
                        is_self: true,
                        is_varargs: false,
                        is_mut: true,
                    }],
                    return_type: option_type,
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![AssociatedTypeDef {
                    name: item_name,
                    bounds: vec![],
                    default: None,
                }],
                is_object_safe: false,
                span,
            };
            self.register_trait(trait_def)
        };

        // IntoIterator trait - enables for loops
        let into_iterator_id = {
            let trait_name = arena.intern_string("IntoIterator");
            let into_iter_method = arena.intern_string("into_iter");
            let into_iter_name = arena.intern_string("IntoIter");
            let id = TypeId::next();

            let item_type = Type::TypeVar(TypeVar::unbound(item_name));
            let into_iter_type = Type::TypeVar(TypeVar::unbound(into_iter_name));

            let trait_def = TraitDef {
                id,
                name: trait_name,
                type_params: vec![],
                super_traits: vec![],
                methods: vec![MethodSig {
                    name: into_iter_method,
                    type_params: vec![],
                    params: vec![ParamDef {
                        name: self_name,
                        ty: self_type,
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    }],
                    return_type: into_iter_type,
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    visibility: Visibility::Public,
                    span,
                    is_extension: false,
                }],
                associated_types: vec![
                    AssociatedTypeDef {
                        name: item_name,
                        bounds: vec![],
                        default: None,
                    },
                    AssociatedTypeDef {
                        name: into_iter_name,
                        bounds: vec![TypeBound::Trait {
                            name: arena.intern_string("Iterator"),
                            args: vec![],
                        }],
                        default: None,
                    },
                ],
                is_object_safe: false,
                span,
            };
            self.register_trait(trait_def)
        };

        BuiltinTraitIds {
            add: add_id,
            sub: sub_id,
            mul: mul_id,
            div: div_id,
            modulo: modulo_id,
            neg: neg_id,
            matmul: matmul_id,
            eq: eq_id,
            ord: ord_id,
            bit_and: bit_and_id,
            bit_or: bit_or_id,
            bit_xor: bit_xor_id,
            not: not_id,
            index: index_id,
            index_mut: index_mut_id,
            display: display_id,
            debug: debug_id,
            clone: clone_id,
            drop: drop_id,
            iterator: iterator_id,
            into_iterator: into_iterator_id,
        }
    }

    /// Helper to register a binary operator trait with associated Output type
    fn register_operator_trait(
        &mut self,
        arena: &mut crate::arena::AstArena,
        trait_name_str: &str,
        rhs_param_name: InternedString,
        output_name: InternedString,
        method_name_str: &str,
        span: &Span,
    ) -> TypeId {
        let trait_name = arena.intern_string(trait_name_str);
        let method_name = arena.intern_string(method_name_str);
        let self_name = arena.intern_string("Self");
        let id = TypeId::next();

        let self_type = Type::TypeVar(TypeVar::unbound(self_name));
        let rhs_type = Type::TypeVar(TypeVar::unbound(rhs_param_name));
        let output_type = Type::TypeVar(TypeVar::unbound(output_name));

        let trait_def = TraitDef {
            id,
            name: trait_name,
            type_params: vec![TypeParam {
                name: rhs_param_name,
                bounds: vec![],
                variance: Variance::Invariant,
                default: Some(self_type.clone()),
                span: *span,
            }],
            super_traits: vec![],
            methods: vec![MethodSig {
                name: method_name,
                type_params: vec![],
                params: vec![
                    ParamDef {
                        name: self_name,
                        ty: self_type,
                        is_self: true,
                        is_varargs: false,
                        is_mut: false,
                    },
                    ParamDef {
                        name: rhs_param_name,
                        ty: rhs_type,
                        is_self: false,
                        is_varargs: false,
                        is_mut: false,
                    },
                ],
                return_type: output_type,
                where_clause: vec![],
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span: *span,
                is_extension: false,
            }],
            associated_types: vec![AssociatedTypeDef {
                name: output_name,
                bounds: vec![],
                default: None,
            }],
            is_object_safe: false, // Associated type makes it non-object-safe
            span: *span,
        };

        self.register_trait(trait_def)
    }

    /// Helper to register a unary operator trait with associated Output type
    fn register_unary_operator_trait(
        &mut self,
        arena: &mut crate::arena::AstArena,
        trait_name_str: &str,
        output_name: InternedString,
        method_name_str: &str,
        span: &Span,
    ) -> TypeId {
        let trait_name = arena.intern_string(trait_name_str);
        let method_name = arena.intern_string(method_name_str);
        let self_name = arena.intern_string("Self");
        let id = TypeId::next();

        let self_type = Type::TypeVar(TypeVar::unbound(self_name));
        let output_type = Type::TypeVar(TypeVar::unbound(output_name));

        let trait_def = TraitDef {
            id,
            name: trait_name,
            type_params: vec![],
            super_traits: vec![],
            methods: vec![MethodSig {
                name: method_name,
                type_params: vec![],
                params: vec![ParamDef {
                    name: self_name,
                    ty: self_type,
                    is_self: true,
                    is_varargs: false,
                    is_mut: false,
                }],
                return_type: output_type,
                where_clause: vec![],
                is_static: false,
                is_async: false,
                visibility: Visibility::Public,
                span: *span,
                is_extension: false,
            }],
            associated_types: vec![AssociatedTypeDef {
                name: output_name,
                bounds: vec![],
                default: None,
            }],
            is_object_safe: false,
            span: *span,
        };

        self.register_trait(trait_def)
    }

    /// Get the trait ID for a binary operator
    pub fn get_operator_trait(&self, op: &str) -> Option<TypeId> {
        let trait_name = match op {
            "+" => "Add",
            "-" => "Sub",
            "*" => "Mul",
            "/" => "Div",
            "%" => "Mod",
            "@" => "MatMul",
            "==" | "!=" => "Eq",
            "<" | ">" | "<=" | ">=" => "Ord",
            "&" => "BitAnd",
            "|" => "BitOr",
            "^" => "BitXor",
            _ => return None,
        };

        let interned = InternedString::new_global(trait_name);
        self.trait_name_to_id.get(&interned).copied()
    }

    /// Get the trait ID for a unary operator
    pub fn get_unary_operator_trait(&self, op: &str) -> Option<TypeId> {
        let trait_name = match op {
            "-" => "Neg",
            "!" => "Not",
            _ => return None,
        };

        let interned = InternedString::new_global(trait_name);
        self.trait_name_to_id.get(&interned).copied()
    }

    /// Get the method name for a binary operator
    pub fn get_operator_method(&self, op: &str) -> Option<&'static str> {
        match op {
            "+" => Some("add"),
            "-" => Some("sub"),
            "*" => Some("mul"),
            "/" => Some("div"),
            "%" => Some("mod"),
            "@" => Some("matmul"),
            "==" => Some("eq"),
            "!=" => Some("ne"),
            "<" => Some("lt"),
            ">" => Some("gt"),
            "<=" => Some("le"),
            ">=" => Some("ge"),
            "&" => Some("bitand"),
            "|" => Some("bitor"),
            "^" => Some("bitxor"),
            _ => None,
        }
    }

    /// Get the method name for a unary operator
    pub fn get_unary_operator_method(&self, op: &str) -> Option<&'static str> {
        match op {
            "-" => Some("neg"),
            "!" => Some("not"),
            _ => None,
        }
    }
}
