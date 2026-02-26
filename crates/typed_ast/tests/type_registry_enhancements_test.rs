//! Test for type registry enhancements from universal type system

use zyntax_typed_ast::arena::{AstArena, InternedString};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::*;

#[test]
fn test_enhanced_type_var_kinds() {
    // Test new TypeVarKind variants
    let integral_var = TypeVar {
        id: TypeVarId::next(),
        name: None,
        kind: TypeVarKind::Integral,
    };
    assert_eq!(integral_var.kind, TypeVarKind::Integral);

    let floating_var = TypeVar {
        id: TypeVarId::next(),
        name: None,
        kind: TypeVarKind::Floating,
    };
    assert_eq!(floating_var.kind, TypeVarKind::Floating);

    let numeric_var = TypeVar {
        id: TypeVarId::next(),
        name: None,
        kind: TypeVarKind::Numeric,
    };
    assert_eq!(numeric_var.kind, TypeVarKind::Numeric);
}

#[test]
fn test_field_def_with_properties() {
    let mut arena = AstArena::new();

    let getter_sig = MethodSig {
        name: arena.intern_string("get_value"),
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::String),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        visibility: Visibility::Public,
        span: Span::new(0, 0),
        is_extension: false,
    };

    let setter_sig = MethodSig {
        name: arena.intern_string("set_value"),
        type_params: vec![],
        params: vec![ParamDef {
            name: arena.intern_string("value"),
            ty: Type::Primitive(PrimitiveType::String),
            is_self: false,
            is_mut: false,
            is_varargs: false,
        }],
        return_type: Type::Primitive(PrimitiveType::Unit),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        visibility: Visibility::Public,
        span: Span::new(0, 0),
        is_extension: false,
    };

    let field = FieldDef {
        name: arena.intern_string("value"),
        ty: Type::Primitive(PrimitiveType::String),
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span: Span::new(0, 0),
        getter: Some(Box::new(getter_sig)),
        setter: Some(Box::new(setter_sig)),
        is_synthetic: true,
    };

    assert!(field.getter.is_some());
    assert!(field.setter.is_some());
    assert!(field.is_synthetic);
}

#[test]
fn test_extension_methods() {
    let mut arena = AstArena::new();

    let ext_method = MethodSig {
        name: arena.intern_string("to_uppercase"),
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::Unit),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        visibility: Visibility::Public,
        span: Span::new(0, 0),
        is_extension: true,
    };

    assert!(ext_method.is_extension);
}

#[test]
fn test_async_kind_enhancements() {
    // Test that Coroutine and Generator variants exist
    let _coroutine = AsyncKind::Coroutine;
    let _generator = AsyncKind::Generator;

    // They should not be equal to other variants
    assert_ne!(AsyncKind::Coroutine, AsyncKind::Async);
    assert_ne!(AsyncKind::Generator, AsyncKind::Async);
}

#[test]
fn test_const_binary_op() {
    // Test binary operations in const values
    let add_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Add,
        left: Box::new(ConstValue::Int(10)),
        right: Box::new(ConstValue::Int(5)),
    };

    match add_expr {
        ConstValue::BinaryOp {
            op: ConstBinaryOp::Add,
            ..
        } => {
            // Success - binary op exists and works
        }
        _ => panic!("Expected BinaryOp"),
    }
}
