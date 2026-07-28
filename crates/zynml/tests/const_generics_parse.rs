//! Parse-shape test for const-generic surface syntax.
//!
//! Verifies the grammar accepts an integer const argument in
//! type-argument position (`Buffer<f32, 4>`) and that the resulting
//! `Type::Named` keeps the integer in `const_args` — separate from the
//! type argument in `type_args`. This is the keystone the ML roadmap's
//! Phase 0 sits on: the backend (`MonomorphizationKey.const_args`,
//! `ConstEvaluator`) already consumes `const_args`; before this the
//! parser hardcoded them empty. No lowering / monomorphization is
//! exercised here — declaration-side `const N: usize` params and the
//! `Foo<T, N>` variable spelling are a separate follow-on.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::typed_ast::{TypedClass, TypedDeclaration, TypedFunction};
use zyntax_typed_ast::{ConstValue, Type};

fn parse(src: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    grammar
        .parse_with_filename(src, "<const_generics_parse>")
        .expect("source should parse")
}

fn return_type_of(program: &zyntax_typed_ast::TypedProgram, name: &str) -> Type {
    function(program, name).return_type.clone()
}

fn function<'a>(program: &'a zyntax_typed_ast::TypedProgram, name: &str) -> &'a TypedFunction {
    for decl in &program.declarations {
        if let TypedDeclaration::Function(f) = &decl.node {
            if f.name.resolve_global().as_deref() == Some(name) {
                return f;
            }
        }
    }
    panic!("function `{name}` not found in parsed program");
}

fn class<'a>(program: &'a zyntax_typed_ast::TypedProgram, name: &str) -> &'a TypedClass {
    for decl in &program.declarations {
        if let TypedDeclaration::Class(c) = &decl.node {
            if c.name.resolve_global().as_deref() == Some(name) {
                return c;
            }
        }
    }
    panic!("struct `{name}` not found in parsed program");
}

#[test]
fn struct_const_param_declaration_is_recognized() {
    let program = parse(
        r#"
        struct Buffer<T, const N: usize> {
            len: i64
        }
        "#,
    );
    let c = class(&program, "Buffer");
    assert_eq!(c.type_params.len(), 2, "T and N are both generic params");
    assert!(!c.type_params[0].is_const, "T is an ordinary type param");
    assert!(
        c.type_params[1].is_const,
        "N must be recognized as a const generic param on the struct"
    );
    assert!(
        c.type_params[1].const_ty.is_some(),
        "the struct const param must carry its declared type"
    );
}

#[test]
fn const_param_declaration_is_recognized() {
    let program = parse(
        r#"
        def make<T, const N: usize>(): i64 {
            return 0
        }
        "#,
    );
    let f = function(&program, "make");
    assert_eq!(f.type_params.len(), 2, "T and N are both generic params");

    let t = &f.type_params[0];
    assert!(!t.is_const, "T must be an ordinary type parameter");
    assert!(t.const_ty.is_none());

    let n = &f.type_params[1];
    assert!(
        n.is_const,
        "N must be recognized as a const generic parameter"
    );
    assert!(
        n.const_ty.is_some(),
        "the const param must carry its declared type (usize)"
    );
}

#[test]
fn integer_const_arg_lands_in_const_args() {
    let program = parse(
        r#"
        def make(): Buffer<f32, 4> {
        }
        "#,
    );
    match return_type_of(&program, "make") {
        Type::Named {
            type_args,
            const_args,
            ..
        } => {
            assert_eq!(
                const_args,
                vec![ConstValue::Int(4)],
                "the `4` in Buffer<f32, 4> must land in const_args"
            );
            assert_eq!(
                type_args.len(),
                1,
                "`f32` must stay a type argument, kept separate from the const"
            );
        }
        other => panic!("expected Type::Named for Buffer<f32, 4>, got {other:?}"),
    }
}

#[test]
fn multiple_const_args_preserve_order() {
    let program = parse(
        r#"
        def make(): Matrix<f32, 3, 4> {
        }
        "#,
    );
    match return_type_of(&program, "make") {
        Type::Named {
            type_args,
            const_args,
            ..
        } => {
            assert_eq!(
                const_args,
                vec![ConstValue::Int(3), ConstValue::Int(4)],
                "both const dims must be captured, in source order"
            );
            assert_eq!(type_args.len(), 1, "only `f32` is a type argument");
        }
        other => panic!("expected Type::Named for Matrix<f32, 3, 4>, got {other:?}"),
    }
}

#[test]
fn purely_typed_generics_still_have_empty_const_args() {
    // Regression guard: a use site with only type arguments must not
    // spuriously gain const args now that the arg list is partitioned.
    let program = parse(
        r#"
        def make(): Pair<f32, i64> {
        }
        "#,
    );
    match return_type_of(&program, "make") {
        Type::Named {
            type_args,
            const_args,
            ..
        } => {
            assert!(
                const_args.is_empty(),
                "no integer args -> const_args must stay empty, got {const_args:?}"
            );
            assert_eq!(
                type_args.len(),
                2,
                "both `f32` and `i64` are type arguments"
            );
        }
        other => panic!("expected Type::Named for Pair<f32, i64>, got {other:?}"),
    }
}
