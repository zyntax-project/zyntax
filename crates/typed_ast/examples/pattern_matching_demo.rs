//! # Advanced Pattern Matching System Demonstration
//!
//! This example showcases the comprehensive pattern matching capabilities
//! of the Zyntax TypedAST, inspired by Rust and Haxe pattern matching systems.

use zyntax_typed_ast::{type_registry::*, typed_ast::*, AstArena, Span};
use zyntax_typed_ast::{AsyncKind, CallingConvention, ConstValue, NullabilityKind};

/// Demonstrates basic pattern matching constructs
fn basic_patterns_demo() {
    let mut arena = AstArena::new();

    println!("🔥 Basic Pattern Matching Patterns:\n");

    // Wildcard pattern: _
    let wildcard = TypedPattern::wildcard();
    println!("✅ Wildcard: _");

    // Variable bindings: x, mut y
    let immutable_var = TypedPattern::immutable_var(arena.intern_string("x"));
    let mutable_var = TypedPattern::mutable_var(arena.intern_string("y"));
    println!("✅ Variables: x (immutable), mut y (mutable)");

    // Literal patterns
    let int_pattern = TypedPattern::int(42);
    let bool_pattern = TypedPattern::bool(true);
    let string_pattern = TypedPattern::string(arena.intern_string("hello"));
    println!("✅ Literals: 42, true, \"hello\"");

    // Tuple patterns: (x, y, _)
    let tuple_pattern = TypedPattern::tuple(vec![
        typed_node(
            immutable_var.clone(),
            Type::Primitive(PrimitiveType::I32),
            Span::new(0, 1),
        ),
        typed_node(
            mutable_var,
            Type::Primitive(PrimitiveType::I32),
            Span::new(3, 4),
        ),
        typed_node(wildcard.clone(), Type::Any, Span::new(6, 7)),
    ]);
    println!("✅ Tuple: (x, mut y, _)");

    assert!(int_pattern.binds_variables() == false);
    assert!(immutable_var.binds_variables() == true);
    assert!(wildcard.is_exhaustive() == true);
}

/// Demonstrates advanced pattern matching features
fn advanced_patterns_demo() {
    let mut arena = AstArena::new();

    println!("\n🚀 Advanced Pattern Matching Features:\n");

    // Range patterns: 1..=10, 'a'..='z'
    let int_range = TypedPattern::range(
        TypedLiteralPattern::Integer(1),
        TypedLiteralPattern::Integer(10),
        true, // inclusive
    );
    println!("✅ Range patterns: 1..=10");

    let char_range = TypedPattern::range(
        TypedLiteralPattern::Char('a'),
        TypedLiteralPattern::Char('z'),
        true,
    );
    println!("✅ Character range: 'a'..='z'");

    // Slice patterns: [head, tail @ ..]
    let slice_pattern = TypedPattern::Slice {
        prefix: vec![typed_node(
            TypedPattern::immutable_var(arena.intern_string("head")),
            Type::Primitive(PrimitiveType::I32),
            Span::new(1, 5),
        )],
        middle: Some(Box::new(typed_node(
            TypedPattern::Rest {
                name: Some(arena.intern_string("tail")),
                mutability: Mutability::Immutable,
            },
            Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                size: None,
                nullability: NullabilityKind::NonNull,
            },
            Span::new(7, 15),
        ))),
        suffix: vec![],
    };
    println!("✅ Slice patterns: [head, tail @ ..]");

    // At patterns: binding @ pattern
    let at_pattern = TypedPattern::at(
        arena.intern_string("opt"),
        Mutability::Immutable,
        typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Option"),
                variant: arena.intern_string("Some"),
                fields: vec![typed_node(
                    TypedPattern::immutable_var(arena.intern_string("x")),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(15, 16),
                )],
            },
            Type::Named {
                id: TypeId::next(), // "Option",
                type_args: vec![Type::Primitive(PrimitiveType::I32)],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            },
            Span::new(10, 20),
        ),
    );
    println!("✅ At patterns: opt @ Some(x)");

    // Reference patterns: &x, &mut y
    let ref_pattern = TypedPattern::reference(
        typed_node(
            TypedPattern::immutable_var(arena.intern_string("x")),
            Type::Primitive(PrimitiveType::I32),
            Span::new(1, 2),
        ),
        Mutability::Immutable,
    );
    println!("✅ Reference patterns: &x, &mut y");
}

/// Demonstrates Rust-style pattern matching
fn rust_style_patterns() {
    let mut arena = AstArena::new();

    println!("\n🦀 Rust-Style Pattern Matching:\n");

    // Option matching: Some(x) | None
    let option_match = TypedMatch {
        scrutinee: Box::new(typed_node(
            TypedExpression::Variable(arena.intern_string("maybe_value")),
            Type::Named {
                id: TypeId::next(), // "Option",
                type_args: vec![Type::Primitive(PrimitiveType::I32)],
                const_args: Vec::new(),
                variance: Vec::new(),
                nullability: NullabilityKind::NonNull,
            },
            Span::new(6, 17),
        )),
        arms: vec![
            TypedMatchArm {
                pattern: Box::new(typed_node(
                    TypedPattern::Enum {
                        name: arena.intern_string("Option"),
                        variant: arena.intern_string("Some"),
                        fields: vec![typed_node(
                            TypedPattern::immutable_var(arena.intern_string("x")),
                            Type::Primitive(PrimitiveType::I32),
                            Span::new(25, 26),
                        )],
                    },
                    Type::Any,
                    Span::new(20, 30),
                )),
                guard: None,
                body: Box::new(typed_node(
                    TypedExpression::Variable(arena.intern_string("x")),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(35, 36),
                )),
            },
            TypedMatchArm {
                pattern: Box::new(typed_node(
                    TypedPattern::Enum {
                        name: arena.intern_string("Option"),
                        variant: arena.intern_string("None"),
                        fields: vec![],
                    },
                    Type::Any,
                    Span::new(40, 44),
                )),
                guard: None,
                body: Box::new(typed_node(
                    TypedExpression::Literal(TypedLiteral::Integer(0)),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(48, 49),
                )),
            },
        ],
    };
    println!("✅ Option matching: Some(x) => x, None => 0");

    // Struct destructuring: Point { x, y }
    let struct_pattern = TypedPattern::Struct {
        name: arena.intern_string("Point"),
        fields: vec![
            TypedFieldPattern {
                name: arena.intern_string("x"),
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(arena.intern_string("px")),
                    Type::Primitive(PrimitiveType::F64),
                    Span::new(15, 17),
                )),
            },
            TypedFieldPattern {
                name: arena.intern_string("y"),
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(arena.intern_string("py")),
                    Type::Primitive(PrimitiveType::F64),
                    Span::new(22, 24),
                )),
            },
        ],
    };
    println!("✅ Struct destructuring: Point {{ x: px, y: py }}");

    // Guards: x if x > 0
    let guard_pattern = TypedPattern::guard(
        typed_node(
            TypedPattern::immutable_var(arena.intern_string("x")),
            Type::Primitive(PrimitiveType::I32),
            Span::new(0, 1),
        ),
        typed_node(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Gt,
                left: Box::new(typed_node(
                    TypedExpression::Variable(arena.intern_string("x")),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(5, 6),
                )),
                right: Box::new(typed_node(
                    TypedExpression::Literal(TypedLiteral::Integer(0)),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(9, 10),
                )),
            }),
            Type::Primitive(PrimitiveType::Bool),
            Span::new(5, 10),
        ),
    );
    println!("✅ Guard patterns: x if x > 0");

    assert_eq!(option_match.arms.len(), 2);
}

/// Demonstrates Haxe-style pattern matching features
fn haxe_style_patterns() {
    let mut arena = AstArena::new();

    println!("\n🎯 Haxe-Style Pattern Matching:\n");

    // Map/Object patterns: { name: n, age: a }
    let map_pattern = TypedPattern::Map(TypedMapPattern {
        entries: vec![
            TypedMapPatternEntry::KeyValue {
                key: typed_node(
                    TypedLiteralPattern::String(arena.intern_string("name")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(2, 8),
                ),
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(arena.intern_string("n")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(10, 11),
                )),
            },
            TypedMapPatternEntry::KeyValue {
                key: typed_node(
                    TypedLiteralPattern::String(arena.intern_string("age")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(13, 18),
                ),
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(arena.intern_string("a")),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(20, 21),
                )),
            },
            TypedMapPatternEntry::Rest {
                name: Some(arena.intern_string("rest")),
                mutability: Mutability::Immutable,
            },
        ],
        exhaustive: false,
    });
    println!("✅ Map patterns: {{ name: n, age: a, ..rest }}");

    // Regex patterns: ~/^\\d+$/
    let regex_pattern = TypedPattern::Regex {
        pattern: arena.intern_string("^\\d+$"),
        flags: Some(arena.intern_string("g")),
    };
    println!("✅ Regex patterns: ~/^\\d+$/g");

    // View patterns (active patterns): view_func => pattern
    let view_pattern = TypedPattern::View {
        view_function: Box::new(typed_node(
            TypedExpression::Variable(arena.intern_string("parse_int")),
            Type::Function {
                params: vec![ParamInfo {
                    name: None,
                    ty: Type::Primitive(PrimitiveType::String),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                }],
                return_type: Box::new(Type::Named {
                    id: TypeId::next(), // "Option",
                    type_args: vec![Type::Primitive(PrimitiveType::I32)],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                }),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: AsyncKind::Sync,
                calling_convention: CallingConvention::Default,
                nullability: NullabilityKind::NonNull,
            },
            Span::new(0, 9),
        )),
        pattern: Box::new(typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Option"),
                variant: arena.intern_string("Some"),
                fields: vec![typed_node(
                    TypedPattern::immutable_var(arena.intern_string("num")),
                    Type::Primitive(PrimitiveType::I32),
                    Span::new(18, 21),
                )],
            },
            Type::Any,
            Span::new(13, 25),
        )),
    };
    println!("✅ View patterns: parse_int => Some(num)");

    // Constructor patterns with type args: Vec<String>(pattern)
    let constructor_pattern = TypedPattern::Constructor {
        constructor: Type::Named {
            id: TypeId::next(), // "Vec",
            type_args: vec![Type::Primitive(PrimitiveType::String)],
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::NonNull,
        },
        pattern: Box::new(typed_node(
            TypedPattern::Array(vec![
                typed_node(
                    TypedPattern::immutable_var(arena.intern_string("first")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(12, 17),
                ),
                typed_node(
                    TypedPattern::Rest {
                        name: Some(arena.intern_string("rest")),
                        mutability: Mutability::Immutable,
                    },
                    Type::Any,
                    Span::new(19, 27),
                ),
            ]),
            Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::String)),
                size: None,
                nullability: NullabilityKind::NonNull,
            },
            Span::new(10, 30),
        )),
    };
    println!("✅ Constructor patterns: Vec<String>([first, ..rest])");
}

/// Demonstrates advanced pattern features
fn advanced_features_demo() {
    let mut arena = AstArena::new();

    println!("\n⭐ Advanced Pattern Features:\n");

    // Or patterns: Red | Green | Blue
    let color_or_pattern = TypedPattern::or(vec![
        typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Color"),
                variant: arena.intern_string("Red"),
                fields: vec![],
            },
            Type::Any,
            Span::new(0, 3),
        ),
        typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Color"),
                variant: arena.intern_string("Green"),
                fields: vec![],
            },
            Type::Any,
            Span::new(6, 11),
        ),
        typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Color"),
                variant: arena.intern_string("Blue"),
                fields: vec![],
            },
            Type::Any,
            Span::new(14, 18),
        ),
    ]);
    println!("✅ Or patterns: Red | Green | Blue");

    // Nested patterns: Some(Ok(value))
    let nested_pattern = TypedPattern::Enum {
        name: arena.intern_string("Option"),
        variant: arena.intern_string("Some"),
        fields: vec![typed_node(
            TypedPattern::Enum {
                name: arena.intern_string("Result"),
                variant: arena.intern_string("Ok"),
                fields: vec![typed_node(
                    TypedPattern::immutable_var(arena.intern_string("value")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(8, 13),
                )],
            },
            Type::Any,
            Span::new(5, 15),
        )],
    };
    println!("✅ Nested patterns: Some(Ok(value))");

    // Error patterns: Error(kind, message)
    let error_pattern = TypedPattern::Error {
        error_type: Some(Type::Named {
            id: TypeId::next(), // "IOError",
            type_args: vec![],
            const_args: Vec::new(),
            variance: Vec::new(),
            nullability: NullabilityKind::NonNull,
        }),
        pattern: Box::new(typed_node(
            TypedPattern::Tuple(vec![
                typed_node(
                    TypedPattern::immutable_var(arena.intern_string("kind")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(12, 16),
                ),
                typed_node(
                    TypedPattern::immutable_var(arena.intern_string("msg")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(18, 21),
                ),
            ]),
            Type::Any,
            Span::new(10, 25),
        )),
    };
    println!("✅ Error patterns: IOError(kind, msg)");

    // Async patterns: async pattern
    let async_pattern = TypedPattern::Async(Box::new(typed_node(
        TypedPattern::immutable_var(arena.intern_string("result")),
        Type::Primitive(PrimitiveType::String),
        Span::new(6, 12),
    )));
    println!("✅ Async patterns: async result");

    // Macro patterns: vec![x, y, z]
    let macro_pattern = TypedPattern::Macro {
        name: arena.intern_string("vec"),
        args: vec![
            typed_node(
                TypedPattern::immutable_var(arena.intern_string("x")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(5, 6),
            ),
            typed_node(
                TypedPattern::immutable_var(arena.intern_string("y")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(8, 9),
            ),
            typed_node(
                TypedPattern::immutable_var(arena.intern_string("z")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(11, 12),
            ),
        ],
    };
    println!("✅ Macro patterns: vec![x, y, z]");
}

/// Demonstrates pattern exhaustiveness checking
fn exhaustiveness_demo() {
    let mut arena = AstArena::new();

    println!("\n🔍 Pattern Exhaustiveness & Analysis:\n");

    let patterns = vec![
        ("Wildcard", TypedPattern::wildcard()),
        (
            "Variable",
            TypedPattern::immutable_var(arena.intern_string("x")),
        ),
        ("Literal", TypedPattern::int(42)),
        (
            "Range",
            TypedPattern::range(
                TypedLiteralPattern::Integer(1),
                TypedLiteralPattern::Integer(10),
                true,
            ),
        ),
    ];

    for (name, pattern) in patterns {
        println!(
            "📋 {}: binds_vars={}, exhaustive={}",
            name,
            pattern.binds_variables(),
            pattern.is_exhaustive()
        );
    }
}

fn main() {
    println!("🚀 Zyntax Advanced Pattern Matching System Demo\n");

    basic_patterns_demo();
    advanced_patterns_demo();
    rust_style_patterns();
    haxe_style_patterns();
    advanced_features_demo();
    exhaustiveness_demo();

    println!("\n✨ Pattern Matching Features Summary:");
    println!("📋 Supported Pattern Types:");
    println!("   • Basic: Wildcard, Variables, Literals, Tuples");
    println!("   • Advanced: Ranges, Slices, Guards, At-patterns");
    println!("   • Rust-style: Structs, Enums, References, Box");
    println!("   • Haxe-style: Maps, Regex, View patterns, Constructors");
    println!("   • Meta: Or-patterns, Nested, Error, Async, Macros");
    println!("   • Analysis: Variable binding, Exhaustiveness checking");
    println!("\n🎯 Ready for advanced pattern matching compilation!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pattern_construction() {
        let mut arena = AstArena::new();

        let wildcard = TypedPattern::wildcard();
        assert!(matches!(wildcard, TypedPattern::Wildcard));
        assert!(!wildcard.binds_variables());
        assert!(wildcard.is_exhaustive());

        let var = TypedPattern::immutable_var(arena.intern_string("test"));
        assert!(var.binds_variables());
        assert!(var.is_exhaustive());

        let int_pattern = TypedPattern::int(42);
        assert!(!int_pattern.binds_variables());
        assert!(!int_pattern.is_exhaustive());
    }

    #[test]
    fn test_range_patterns() {
        let range = TypedPattern::range(
            TypedLiteralPattern::Integer(1),
            TypedLiteralPattern::Integer(10),
            true,
        );

        if let TypedPattern::Range { inclusive, .. } = range {
            assert!(inclusive);
        } else {
            panic!("Expected range pattern");
        }
    }

    #[test]
    fn test_or_patterns() {
        let mut arena = AstArena::new();

        let or_pattern = TypedPattern::or(vec![
            typed_node(
                TypedPattern::int(1),
                Type::Primitive(PrimitiveType::I32),
                Span::new(0, 1),
            ),
            typed_node(
                TypedPattern::int(2),
                Type::Primitive(PrimitiveType::I32),
                Span::new(4, 5),
            ),
            typed_node(
                TypedPattern::int(3),
                Type::Primitive(PrimitiveType::I32),
                Span::new(8, 9),
            ),
        ]);

        if let TypedPattern::Or(patterns) = or_pattern {
            assert_eq!(patterns.len(), 3);
        } else {
            panic!("Expected or pattern");
        }
    }

    #[test]
    fn test_guard_patterns() {
        let mut arena = AstArena::new();

        let guard = TypedPattern::guard(
            typed_node(
                TypedPattern::immutable_var(arena.intern_string("x")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(0, 1),
            ),
            typed_node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                Type::Primitive(PrimitiveType::Bool),
                Span::new(5, 9),
            ),
        );

        assert!(guard.binds_variables());
        if let TypedPattern::Guard {
            pattern,
            condition: _,
        } = guard
        {
            assert!(pattern.node.binds_variables());
        } else {
            panic!("Expected guard pattern");
        }
    }

    #[test]
    fn test_slice_patterns() {
        let mut arena = AstArena::new();

        let slice = TypedPattern::Slice {
            prefix: vec![typed_node(
                TypedPattern::immutable_var(arena.intern_string("first")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(1, 6),
            )],
            middle: Some(Box::new(typed_node(
                TypedPattern::Rest {
                    name: Some(arena.intern_string("middle")),
                    mutability: Mutability::Immutable,
                },
                Type::Array {
                    element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                    size: None,
                    nullability: NullabilityKind::NonNull,
                },
                Span::new(7, 16),
            ))),
            suffix: vec![typed_node(
                TypedPattern::immutable_var(arena.intern_string("last")),
                Type::Primitive(PrimitiveType::I32),
                Span::new(17, 21),
            )],
        };

        assert!(slice.binds_variables());
        if let TypedPattern::Slice {
            prefix,
            middle,
            suffix,
        } = slice
        {
            assert_eq!(prefix.len(), 1);
            assert!(middle.is_some());
            assert_eq!(suffix.len(), 1);
        } else {
            panic!("Expected slice pattern");
        }
    }

    #[test]
    fn test_map_patterns() {
        let mut arena = AstArena::new();

        let map = TypedPattern::Map(TypedMapPattern {
            entries: vec![TypedMapPatternEntry::KeyValue {
                key: typed_node(
                    TypedLiteralPattern::String(arena.intern_string("key")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(2, 7),
                ),
                pattern: Box::new(typed_node(
                    TypedPattern::immutable_var(arena.intern_string("value")),
                    Type::Primitive(PrimitiveType::String),
                    Span::new(9, 14),
                )),
            }],
            exhaustive: false,
        });

        assert!(map.binds_variables());
    }
}
