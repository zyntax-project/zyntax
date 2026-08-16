#![cfg(feature = "cranelift-backend")]

//! Loop-carried locals through typed-AST lowering.
//!
//! The existing executing loop tests build `HirFunction` by hand with
//! explicit phi nodes, so they prove the backend handles loop-carried
//! values — never that lowering *creates* those phis. This one goes
//! through `compile_to_hir` and executes, which is the untested edge.
//!
//! ```text
//! fn count_up() -> i32 {
//!     let i = 0
//!     while i < 3 { i = i + 1 }
//!     return i
//! }
//! ```

use std::sync::Arc;

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::{compile_to_hir, CompilationConfig};
use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::type_registry::Mutability;
use zyntax_typed_ast::typed_ast::{TypedBinary, TypedBlock, TypedLet, TypedWhile};
use zyntax_typed_ast::{
    BinaryOp, CallingConvention, PrimitiveType, Span, Type, TypeRegistry, TypedDeclaration,
    TypedExpression, TypedFunction, TypedLiteral, TypedNode, TypedProgram, TypedStatement,
    Visibility,
};

fn span() -> Span {
    Span::new(0, 10)
}

fn node<T>(inner: T, ty: Type) -> TypedNode<T> {
    TypedNode {
        node: inner,
        ty,
        span: span(),
    }
}

fn i32_ty() -> Type {
    Type::Primitive(PrimitiveType::I32)
}

fn int_lit(v: i128) -> TypedNode<TypedExpression> {
    node(TypedExpression::Literal(TypedLiteral::Integer(v)), i32_ty())
}

fn var(name: zyntax_typed_ast::InternedString) -> TypedNode<TypedExpression> {
    node(TypedExpression::Variable(name), i32_ty())
}

fn binary(
    op: BinaryOp,
    l: TypedNode<TypedExpression>,
    r: TypedNode<TypedExpression>,
    ty: Type,
) -> TypedNode<TypedExpression> {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }),
        ty,
    )
}

/// `let i = 0; while i < 3 { i = i + 1 }; return i`
fn count_up_body(i: zyntax_typed_ast::InternedString) -> TypedBlock {
    let let_i = node(
        TypedStatement::Let(TypedLet {
            name: i,
            ty: i32_ty(),
            mutability: Mutability::Mutable,
            initializer: Some(Box::new(int_lit(0))),
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
    );

    // i = i + 1
    let incr = node(
        TypedStatement::Expression(Box::new(binary(
            BinaryOp::Assign,
            var(i),
            binary(BinaryOp::Add, var(i), int_lit(1), i32_ty()),
            i32_ty(),
        ))),
        Type::Primitive(PrimitiveType::Unit),
    );

    let while_stmt = node(
        TypedStatement::While(TypedWhile {
            condition: Box::new(binary(
                BinaryOp::Lt,
                var(i),
                int_lit(3),
                Type::Primitive(PrimitiveType::Bool),
            )),
            body: TypedBlock {
                statements: vec![incr],
                span: span(),
            },
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
    );

    let ret = node(
        TypedStatement::Return(Some(Box::new(var(i)))),
        Type::Primitive(PrimitiveType::Unit),
    );

    TypedBlock {
        statements: vec![let_i, while_stmt, ret],
        span: span(),
    }
}

fn program(arena: &mut AstArena) -> (TypedProgram, zyntax_typed_ast::InternedString) {
    let i = arena.intern_string("i");
    let name = arena.intern_string("count_up");
    let function = TypedFunction {
        type_params: vec![],
        name,
        params: vec![],
        return_type: i32_ty(),
        body: Some(count_up_body(i)),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        with_handlers: vec![],
        is_pure: false,
        module: None,
    };
    (
        TypedProgram {
            language: None,
            declarations: vec![node(
                TypedDeclaration::Function(function),
                Type::Primitive(PrimitiveType::Unit),
            )],
            span: span(),
            source_files: vec![],
            type_registry: TypeRegistry::new(),
        },
        i,
    )
}

/// A counter written in a `while` body must be visible to the header on
/// the next iteration and after the loop. Lowered, compiled, and run —
/// the phi has to come from lowering, not from the test.
#[test]
fn a_loop_carried_local_survives_lowering() {
    let mut arena = AstArena::new();
    let (mut prog, _) = program(&mut arena);
    let registry = Arc::new(TypeRegistry::new());

    let module = compile_to_hir(
        &mut prog,
        registry,
        CompilationConfig {
            opt_level: 0,
            debug_info: false,
            enable_monomorphization: true,
            ..Default::default()
        },
    )
    .expect("count_up should lower");

    let func = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("count_up"))
        .expect("count_up should be present")
        .clone();

    // The loop header must carry `i` — without a phi there the counter
    // resolves to its pre-loop definition forever.
    let phi_count: usize = func.blocks.values().map(|b| b.phis.len()).sum();

    let mut backend = CraneliftBackend::new().expect("backend");
    backend
        .compile_function(func.id, &func)
        .expect("count_up should compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(func.id).expect("fn ptr");
    let f: extern "C" fn() -> i32 = unsafe { std::mem::transmute(ptr) };
    let got = f();

    assert_eq!(
        got, 3,
        "counter should reach 3 (loop-header phis found: {phi_count})"
    );
}
