//! # Generics Support Demonstration
//!
//! This example shows how the Zyntax TypedAST supports comprehensive generics
//! with various type constraint types from different programming languages.

use zyntax_typed_ast::{
    typed_ast::*, AstArena, AsyncKind, CallingConvention, Lifetime, Mutability, NullabilityKind,
    PrimitiveType, Span, Type, TypeId, TypeVar, TypeVarId, TypeVarKind, Visibility,
};

/// Demonstrates Rust-style generic constraints
fn rust_style_generics() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");
    let u_param = arena.intern_string("U");

    // Generic function: fn process<T: Clone + Send + Sync + 'static>(value: T) -> T
    let rust_function = TypedFunction {
        name: arena.intern_string("process"),
        params: vec![TypedParameter {
            name: arena.intern_string("value"),
            ty: Type::TypeVar(TypeVar::unbound(t_param)),
            mutability: Mutability::Immutable,
            span: Span::new(50, 55),
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
        }],
        return_type: Type::TypeVar(TypeVar::unbound(t_param)),
        body: Some(TypedBlock {
            statements: vec![],
            span: Span::new(70, 80),
        }),
        visibility: Visibility::Public,
        is_async: false,
        ..Default::default()
    };

    // Type parameters with Rust-style bounds
    let type_params = vec![TypedTypeParam {
        name: t_param,
        bounds: vec![
            TypedTypeBound::Trait(Type::Named {
                id: TypeId::next(), // "Clone",
                type_args: vec![],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            }),
            TypedTypeBound::Send,
            TypedTypeBound::Sync,
            TypedTypeBound::Static,
        ],
        default: None,
        span: Span::new(15, 45),
    }];

    println!("✅ Rust-style generics: T: Clone + Send + Sync + 'static");
}

/// Demonstrates Java-style generic constraints
fn java_style_generics() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");
    let u_param = arena.intern_string("U");

    // Java: class Container<T extends Comparable<T> & Serializable>
    let java_class = TypedClass {
        name: arena.intern_string("Container"),
        type_params: vec![TypedTypeParam {
            name: t_param,
            bounds: vec![
                // T extends Comparable<T>
                TypedTypeBound::Subtype(Type::Named {
                    id: TypeId::next(), // "Comparable",
                    type_args: vec![Type::TypeVar(TypeVar::unbound(t_param))],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
                // T extends Serializable
                TypedTypeBound::Subtype(Type::Named {
                    id: TypeId::next(), // "Serializable",
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
            ],
            default: None,
            span: Span::new(15, 55),
        }],
        extends: None,
        implements: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        span: Span::new(0, 100),
    };

    println!("✅ Java-style generics: T extends Comparable<T> & Serializable");
}

/// Demonstrates C#-style generic constraints
fn csharp_style_generics() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");
    let u_param = arena.intern_string("U");

    // C#: class Repository<T> where T : class, IEntity, new()
    let csharp_class = TypedClass {
        name: arena.intern_string("Repository"),
        type_params: vec![TypedTypeParam {
            name: t_param,
            bounds: vec![
                // where T : class (reference type constraint)
                TypedTypeBound::ReferenceType,
                // where T : IEntity
                TypedTypeBound::Trait(Type::Named {
                    id: TypeId::next(), // "IEntity",
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
                // where T : new() (constructor constraint)
                TypedTypeBound::Constructor(vec![]),
            ],
            default: None,
            span: Span::new(15, 20),
        }],
        extends: None,
        implements: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        span: Span::new(0, 100),
    };

    // C#: struct ValueProcessor<T> where T : struct, IComparable<T>
    let value_processor_params = vec![TypedTypeParam {
        name: t_param,
        bounds: vec![
            // where T : struct (value type constraint)
            TypedTypeBound::ValueType,
            // where T : IComparable<T>
            TypedTypeBound::Trait(Type::Named {
                id: TypeId::next(), // "IComparable",
                type_args: vec![Type::TypeVar(TypeVar::unbound(t_param))],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            }),
        ],
        default: None,
        span: Span::new(25, 30),
    }];

    println!("✅ C#-style generics: T : class, IEntity, new() and T : struct, IComparable<T>");
}

/// Demonstrates TypeScript-style generic constraints
fn typescript_style_generics() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");
    let k_param = arena.intern_string("K");

    // TypeScript: function pick<T, K extends keyof T>(obj: T, keys: K[]): T[K]
    let typescript_function = TypedFunction {
        name: arena.intern_string("pick"),
        params: vec![
            TypedParameter {
                name: arena.intern_string("obj"),
                ty: Type::TypeVar(TypeVar::unbound(t_param)),
                mutability: Mutability::Immutable,
                span: Span::new(40, 43),
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
            },
            TypedParameter {
                name: arena.intern_string("keys"),
                ty: Type::Array {
                    element_type: Box::new(Type::TypeVar(TypeVar::unbound(k_param))),
                    size: None,
                    nullability: NullabilityKind::NonNull,
                },
                mutability: Mutability::Immutable,
                span: Span::new(47, 52),
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
            },
        ],
        return_type: Type::Index {
            base: Box::new(Type::TypeVar(TypeVar::unbound(t_param))),
            index: Box::new(Type::TypeVar(TypeVar::unbound(k_param))),
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: Span::new(70, 80),
        }),
        visibility: Visibility::Public,
        is_async: false,
        ..Default::default()
    };

    let type_params = vec![
        TypedTypeParam {
            name: t_param,
            bounds: vec![], // T has no constraints
            default: None,
            span: Span::new(15, 16),
        },
        TypedTypeParam {
            name: k_param,
            bounds: vec![
                // K extends keyof T (using custom constraint)
                TypedTypeBound::Custom {
                    name: arena.intern_string("keyof"),
                    args: vec![Type::TypeVar(TypeVar::unbound(t_param))],
                },
            ],
            default: None,
            span: Span::new(18, 30),
        },
    ];

    println!("✅ TypeScript-style generics: K extends keyof T");
}

/// Demonstrates Haxe-style generic constraints
fn haxe_style_generics() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");

    // Haxe: class DataProcessor<T:(Iterable<String> & {function process():Void})>
    let haxe_class = TypedClass {
        name: arena.intern_string("DataProcessor"),
        type_params: vec![TypedTypeParam {
            name: t_param,
            bounds: vec![
                // T implements Iterable<String>
                TypedTypeBound::Trait(Type::Named {
                    id: TypeId::next(), // "Iterable",
                    type_args: vec![Type::Primitive(PrimitiveType::String)],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
                // T has method process():Void (structural constraint)
                TypedTypeBound::Custom {
                    name: arena.intern_string("structural"),
                    args: vec![Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                        is_varargs: false,
                        has_named_params: false,
                        has_default_params: false,
                        async_kind: AsyncKind::Sync,
                        calling_convention: CallingConvention::Default,
                        nullability: NullabilityKind::NonNull,
                    }],
                },
            ],
            default: None,
            span: Span::new(20, 70),
        }],
        extends: None,
        implements: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        span: Span::new(0, 100),
    };

    println!("✅ Haxe-style generics: T:(Iterable<String> & structural constraints)");
}

/// Demonstrates advanced constraint combinations
fn advanced_constraint_combinations() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");
    let u_param = arena.intern_string("U");
    let lifetime_a = Lifetime::named(arena.intern_string("a"));

    // Complex constraint: T: Clone + Send + 'static + Into<U> where U: Display
    let complex_params = vec![
        TypedTypeParam {
            name: t_param,
            bounds: vec![
                TypedTypeBound::Trait(Type::Named {
                    id: TypeId::next(), // "Clone",
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
                TypedTypeBound::Send,
                TypedTypeBound::Static,
                TypedTypeBound::Trait(Type::Named {
                    id: TypeId::next(), // "Into",
                    type_args: vec![Type::TypeVar(TypeVar::unbound(u_param))],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
            ],
            default: None,
            span: Span::new(10, 40),
        },
        TypedTypeParam {
            name: u_param,
            bounds: vec![TypedTypeBound::Trait(Type::Named {
                id: TypeId::next(), // "Display",
                type_args: vec![],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            })],
            default: None,
            span: Span::new(50, 60),
        },
    ];

    println!("✅ Advanced constraints: T: Clone + Send + 'static + Into<U> where U: Display");
}

/// Demonstrates associated type constraints
fn associated_type_constraints() {
    let mut arena = AstArena::new();
    let t_param = arena.intern_string("T");

    // Rust: fn process<T>() where T: Iterator<Item = String>
    let associated_type_param = TypedTypeParam {
        name: t_param,
        bounds: vec![
            TypedTypeBound::Trait(Type::Named {
                id: TypeId::next(), // "Iterator",
                type_args: vec![],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            }),
            // Associated type constraint: Item = String
            TypedTypeBound::Equality(Type::Primitive(PrimitiveType::String)),
        ],
        default: None,
        span: Span::new(15, 45),
    };

    println!("✅ Associated type constraints: T: Iterator<Item = String>");
}

fn main() {
    println!("🚀 Zyntax TypedAST Generics Support Demonstration\n");

    rust_style_generics();
    java_style_generics();
    csharp_style_generics();
    typescript_style_generics();
    haxe_style_generics();
    advanced_constraint_combinations();
    associated_type_constraints();

    println!("\n✨ All generic constraint types are fully supported!");
    println!("📋 Supported constraint types:");
    println!("   • Trait bounds (T: Display)");
    println!("   • Lifetime bounds (T: 'static)");
    println!("   • Equality constraints (T::Item = String)");
    println!("   • Subtype constraints (T extends Base)");
    println!("   • Supertype constraints (T :> Derived)");
    println!("   • Constructor constraints (T: new())");
    println!("   • Rust constraints (Sized, Copy, Send, Sync)");
    println!("   • C# constraints (ValueType, ReferenceType, Unmanaged)");
    println!("   • Custom constraints with parameters");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_generic_bounds() {
        let mut arena = AstArena::new();
        let t_param = arena.intern_string("T");

        let param = TypedTypeParam {
            name: t_param,
            bounds: vec![
                TypedTypeBound::Send,
                TypedTypeBound::Sync,
                TypedTypeBound::Sized,
            ],
            default: None,
            span: Span::new(0, 20),
        };

        assert_eq!(param.bounds.len(), 3);
        assert!(matches!(param.bounds[0], TypedTypeBound::Send));
        assert!(matches!(param.bounds[1], TypedTypeBound::Sync));
        assert!(matches!(param.bounds[2], TypedTypeBound::Sized));
    }

    #[test]
    fn test_csharp_generic_bounds() {
        let mut arena = AstArena::new();
        let t_param = arena.intern_string("T");

        let param = TypedTypeParam {
            name: t_param,
            bounds: vec![
                TypedTypeBound::ReferenceType,
                TypedTypeBound::Constructor(vec![]),
            ],
            default: None,
            span: Span::new(0, 20),
        };

        assert_eq!(param.bounds.len(), 2);
        assert!(matches!(param.bounds[0], TypedTypeBound::ReferenceType));
        assert!(matches!(param.bounds[1], TypedTypeBound::Constructor(_)));
    }

    #[test]
    fn test_custom_constraint() {
        let mut arena = AstArena::new();
        let t_param = arena.intern_string("T");
        let constraint_name = arena.intern_string("keyof");

        let param = TypedTypeParam {
            name: t_param,
            bounds: vec![TypedTypeBound::Custom {
                name: constraint_name,
                args: vec![Type::Primitive(PrimitiveType::String)],
            }],
            default: None,
            span: Span::new(0, 20),
        };

        if let TypedTypeBound::Custom { name, args } = &param.bounds[0] {
            assert_eq!(*name, constraint_name);
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected custom constraint");
        }
    }
}
