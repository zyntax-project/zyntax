//! Tests for the extended TypedAST features

use zyntax_typed_ast::{
    ast_convert::*, typed_ast::*, AstArena, InternedString, Mutability, PrimitiveType, Span, Type,
    TypeId, TypeRegistry, Visibility,
};
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_class_declaration() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let class_name = arena.intern_string("Person");
    let field_name = arena.intern_string("name");
    let method_name = arena.intern_string("get_name");

    let class = TypedClass {
        name: class_name,
        type_params: vec![],
        extends: None,
        implements: vec![],
        fields: vec![TypedField {
            name: field_name,
            ty: Type::Primitive(PrimitiveType::String),
            initializer: None,
            visibility: Visibility::Private,
            mutability: Mutability::Immutable,
            is_static: false,
            span: Span::new(20, 35),
        }],
        methods: vec![TypedMethod {
            name: method_name,
            type_params: vec![],
            params: vec![TypedMethodParam {
                name: arena.intern_string("self"),
                ty: Type::Named {
                    id: TypeId::next(), // Placeholder type ID for Person class
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                },
                mutability: Mutability::Immutable,
                is_self: true,
                span: Span::new(50, 54),
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::String),
            body: None,
            visibility: Visibility::Public,
            is_static: false,
            is_async: false,
            is_override: false,
            span: Span::new(40, 80),
        }],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: false,
        span: Span::new(0, 100),
    };

    assert_eq!(class.name, class_name);
    assert_eq!(class.fields.len(), 1);
    assert_eq!(class.methods.len(), 1);
    assert_eq!(class.fields[0].mutability, Mutability::Immutable);
    assert_eq!(class.methods[0].visibility, Visibility::Public);
}

#[test]
fn test_enum_with_variants() {
    let mut arena = AstArena::new();
    let enum_name = arena.intern_string("Color");
    let red = arena.intern_string("Red");
    let green = arena.intern_string("Green");
    let blue = arena.intern_string("Blue");

    let color_enum = TypedEnum {
        name: enum_name,
        type_params: vec![],
        variants: vec![
            TypedVariant {
                name: red,
                fields: TypedVariantFields::Unit,
                discriminant: None,
                span: Span::new(10, 15),
            },
            TypedVariant {
                name: green,
                fields: TypedVariantFields::Unit,
                discriminant: None,
                span: Span::new(20, 25),
            },
            TypedVariant {
                name: blue,
                fields: TypedVariantFields::Tuple(vec![
                    Type::Primitive(PrimitiveType::U8),
                    Type::Primitive(PrimitiveType::U8),
                    Type::Primitive(PrimitiveType::U8),
                ]),
                discriminant: None,
                span: Span::new(30, 50),
            },
        ],
        visibility: Visibility::Public,
        span: Span::new(0, 60),
    };

    assert_eq!(color_enum.variants.len(), 3);
    assert!(matches!(
        color_enum.variants[0].fields,
        TypedVariantFields::Unit
    ));
    assert!(matches!(
        color_enum.variants[2].fields,
        TypedVariantFields::Tuple(_)
    ));
}

#[test]
fn test_interface_with_methods() {
    let mut arena = AstArena::new();
    let interface_name = arena.intern_string("Drawable");
    let draw_method = arena.intern_string("draw");

    let drawable = TypedInterface {
        name: interface_name,
        type_params: vec![],
        extends: vec![],
        methods: vec![TypedMethodSignature {
            name: draw_method,
            type_params: vec![],
            params: vec![TypedMethodParam {
                name: arena.intern_string("self"),
                ty: Type::Named {
                    id: TypeId::next(), // Placeholder type ID for Printable interface
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                },
                mutability: Mutability::Immutable,
                is_self: true,
                span: Span::new(15, 19),
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
            }],
            return_type: Type::Primitive(PrimitiveType::Unit),
            is_static: false,
            is_async: false,
            span: Span::new(10, 30),
        }],
        associated_types: vec![],
        visibility: Visibility::Public,
        span: Span::new(0, 40),
    };

    assert_eq!(drawable.methods.len(), 1);
    assert_eq!(drawable.methods[0].params[0].is_self, true);
}

#[test]
fn test_pattern_matching() {
    let mut arena = AstArena::new();
    let var_name = arena.intern_string("x");
    let span = Span::new(0, 10);

    // Test different pattern types
    let wildcard = typed_node(TypedPattern::Wildcard, Type::Any, span);

    let identifier = typed_node(
        TypedPattern::Identifier {
            name: var_name,
            mutability: Mutability::Mutable,
        },
        Type::Primitive(PrimitiveType::I32),
        span,
    );

    let literal = typed_node(
        TypedPattern::Literal(TypedLiteralPattern::Integer(42)),
        Type::Primitive(PrimitiveType::I32),
        span,
    );

    assert!(matches!(wildcard.node, TypedPattern::Wildcard));
    assert!(matches!(identifier.node, TypedPattern::Identifier { .. }));
    assert!(matches!(literal.node, TypedPattern::Literal(_)));
}

#[test]
fn test_lambda_expression() {
    let mut arena = AstArena::new();
    let param_name = arena.intern_string("x");
    let _span = Span::new(0, 20);

    let lambda = TypedLambda {
        params: vec![TypedLambdaParam {
            name: param_name,
            ty: Some(Type::Primitive(PrimitiveType::I32)),
        }],
        body: TypedLambdaBody::Expression(Box::new(typed_node(
            TypedExpression::Variable(param_name),
            Type::Primitive(PrimitiveType::I32),
            Span::new(15, 16),
        ))),
        captures: vec![],
    };

    assert_eq!(lambda.params.len(), 1);
    assert!(lambda.params[0].ty.is_some());
    assert!(lambda.captures.is_empty());
}

#[test]
fn test_ast_conversion_context() {
    struct TestInterner {
        arena: AstArena,
    }

    impl StringInterner for TestInterner {
        fn intern(&mut self, s: &str) -> InternedString {
            self.arena.intern_string(s)
        }
    }

    let type_registry = Box::new(TypeRegistry::new());
    let interner = Box::new(TestInterner {
        arena: AstArena::new(),
    });
    let mut ctx = ConversionContext::new(type_registry, interner);

    // Test module path tracking
    let module_name = ctx.strings.intern("std");
    ctx.enter_module(module_name);
    assert_eq!(ctx.module_path.len(), 1);

    ctx.leave_module();
    assert_eq!(ctx.module_path.len(), 0);

    // Test type caching
    let key = "MyType".to_string();
    let ty = Type::Primitive(PrimitiveType::I32);
    ctx.cache_type(key.clone(), ty.clone());

    assert_eq!(ctx.get_cached_type(&key), Some(&ty));
}

#[test]
fn test_comprehensive_program_structure() {
    let mut arena = AstArena::new();
    let main_func = arena.intern_string("main");
    let person_class = arena.intern_string("Person");

    // Create a comprehensive program with multiple declaration types
    let program = TypedProgram {
        declarations: vec![
            // Import declaration
            typed_node(
                TypedDeclaration::Import(TypedImport {
                    module_path: vec![arena.intern_string("std"), arena.intern_string("io")],
                    items: vec![TypedImportItem::Named {
                        name: arena.intern_string("println"),
                        alias: None,
                    }],
                    span: Span::new(0, 20),
                }),
                Type::Primitive(PrimitiveType::Unit),
                Span::new(0, 20),
            ),
            // Class declaration
            typed_node(
                TypedDeclaration::Class(TypedClass {
                    name: person_class,
                    type_params: vec![],
                    extends: None,
                    implements: vec![],
                    fields: vec![],
                    methods: vec![],
                    constructors: vec![],
                    visibility: Visibility::Public,
                    is_abstract: false,
                    is_final: false,
                    span: Span::new(25, 60),
                }),
                Type::Named {
                    id: TypeId::next(), // Placeholder type ID for Person class
                    type_args: vec![],
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: NullabilityKind::NonNull,
                },
                Span::new(25, 60),
            ),
            // Function declaration
            typed_node(
                TypedDeclaration::Function(TypedFunction {
                    name: main_func,
                    params: vec![],
                    type_params: vec![],
                    return_type: Type::Primitive(PrimitiveType::Unit),
                    body: Some(TypedBlock {
                        statements: vec![],
                        span: Span::new(80, 90),
                    }),
                    visibility: Visibility::Public,
                    is_async: false,
                    is_external: false,
                    calling_convention: CallingConvention::Default,
                    link_name: None,
                    ..Default::default()
                }),
                Type::Function {
                    params: vec![],
                    return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: AsyncKind::Sync,
                    calling_convention: CallingConvention::Default,
                    nullability: NullabilityKind::NonNull,
                },
                Span::new(65, 95),
            ),
        ],
        span: Span::new(0, 100),
        ..Default::default()
    };

    assert_eq!(program.declarations.len(), 3);

    // Verify different declaration types
    assert!(matches!(
        program.declarations[0].node,
        TypedDeclaration::Import(_)
    ));
    assert!(matches!(
        program.declarations[1].node,
        TypedDeclaration::Class(_)
    ));
    assert!(matches!(
        program.declarations[2].node,
        TypedDeclaration::Function(_)
    ));
}
