//! Regression coverage for the lambda-body-drop bug
//! (see `ZYNTAX_LAMBDA_BODY_BUG.md`).
//!
//! Before the fix in `ssa.rs::translate_closure`, closure bodies whose
//! only content was a function or method call silently compiled to a
//! no-op that returned `Constant::I32(0)`. The mini-translator
//! `translate_lambda_expr` only knew about `Literal` / `Variable` /
//! `Binary`, and the `TypedLambdaBody::Block` arm bypassed even that —
//! it emitted a zero directly.
//!
//! These tests build a `TypedProgram` containing a closure whose body
//! is a `Call`, lower it through the real `LoweringContext`, then walk
//! the resulting HIR to assert the closure function's entry block
//! contains an actual `HirInstruction::Call`. A failing run produces a
//! closure whose entry block has zero `Call` instructions — that's
//! the silent-drop bug's signature.
//!
//! The bug was a correctness gap at SSA lowering time, so the test
//! lives in the compiler crate's integration tests; it doesn't need a
//! grammar parser, runtime, or backend.

use std::sync::{Arc, Mutex};

use zyntax_compiler::{
    hir::HirInstruction,
    lowering::{AstLowering, LoweringConfig, LoweringContext},
};
use zyntax_typed_ast::{
    arena::AstArena,
    typed_ast::{TypedBlock, TypedLambda, TypedLambdaBody, TypedLet, TypedParameter},
    typed_node, PrimitiveType, Span, Type, TypeRegistry, TypedCall, TypedDeclaration,
    TypedExpression, TypedFunction, TypedLiteral, TypedProgram, TypedStatement,
};

fn span() -> Span {
    Span::new(0, 1)
}

/// Build `extern fn sink(value: i32): i32`.
fn make_extern_sink(arena: &mut AstArena) -> TypedDeclaration {
    let name = arena.intern_string("sink");
    let value_name = arena.intern_string("value");
    let mut param = TypedParameter::default();
    param.name = value_name;
    param.ty = Type::Primitive(PrimitiveType::I32);
    param.span = span();

    let mut sink_fn = TypedFunction::default();
    sink_fn.name = name;
    sink_fn.params = vec![param];
    sink_fn.return_type = Type::Primitive(PrimitiveType::I32);
    sink_fn.body = None;
    sink_fn.is_external = true;
    TypedDeclaration::Function(sink_fn)
}

fn lit_i32(value: i32) -> Box<zyntax_typed_ast::TypedNode<TypedExpression>> {
    Box::new(typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(value as i128)),
        Type::Primitive(PrimitiveType::I32),
        span(),
    ))
}

/// `sink(arg)` as a TypedExpression::Call. Callee resolution happens
/// later in lowering; we just emit a `Variable` reference here.
fn call_sink(arena: &mut AstArena, arg: i32) -> zyntax_typed_ast::TypedNode<TypedExpression> {
    let sink_name = arena.intern_string("sink");
    typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(
                TypedExpression::Variable(sink_name),
                // The lowering reads through `convert_type`; using a
                // simple primitive here avoids the heavyweight
                // Type::Function shape (which has many required
                // fields we don't care about for this test).
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )),
            type_args: vec![],
            positional_args: vec![*lit_i32(arg)],
            named_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I32),
        span(),
    )
}

/// Wrap a closure value into a `let f = <closure>` statement.
fn let_f_eq_closure(
    arena: &mut AstArena,
    body: TypedLambdaBody,
) -> zyntax_typed_ast::TypedNode<TypedStatement> {
    let lambda_expr = TypedExpression::Lambda(TypedLambda {
        params: vec![],
        body,
        captures: vec![],
    });
    let f_name = arena.intern_string("f");
    typed_node(
        TypedStatement::Let(TypedLet {
            name: f_name,
            // SSA's `translate_closure` infers `(): I64` when the closure's
            // type isn't `Type::Function`; we just need this field to
            // type-check structurally.
            ty: Type::Primitive(PrimitiveType::Unit),
            mutability: zyntax_typed_ast::Mutability::Immutable,
            initializer: Some(Box::new(typed_node(
                lambda_expr,
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ))),
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    )
}

fn return_zero() -> zyntax_typed_ast::TypedNode<TypedStatement> {
    typed_node(
        TypedStatement::Return(Some(lit_i32(0))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    )
}

/// Build a complete program containing `sink` + a `main` whose body
/// holds the given closure body via `let f = def(): <closure_body>`.
fn build_program(closure_body: TypedLambdaBody, arena: &mut AstArena) -> TypedProgram {
    let sink_decl = make_extern_sink(arena);
    let let_stmt = let_f_eq_closure(arena, closure_body);
    let main_name = arena.intern_string("main");

    let mut main_fn = TypedFunction::default();
    main_fn.name = main_name;
    main_fn.return_type = Type::Primitive(PrimitiveType::I32);
    main_fn.body = Some(TypedBlock {
        statements: vec![let_stmt, return_zero()],
        span: span(),
    });

    TypedProgram {
        declarations: vec![
            typed_node(sink_decl, Type::Primitive(PrimitiveType::Unit), span()),
            typed_node(
                TypedDeclaration::Function(main_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    }
}

/// Lower a program through `LoweringContext`. Type-checking is
/// disabled for the synthetic input — the test exercises SSA
/// lowering, not the type checker.
fn lower(mut program: TypedProgram, mut arena: AstArena) -> zyntax_compiler::hir::HirModule {
    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("closure_test");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let module = ctx
        .lower_program(&mut program)
        .expect("lower program containing closure");
    std::env::remove_var("SKIP_TYPE_CHECK");
    module
}

/// Find the synthesised closure function in a lowered module. SSA
/// names lambdas with a `__lambda_` prefix.
fn find_closure_fn(module: &zyntax_compiler::hir::HirModule) -> &zyntax_compiler::hir::HirFunction {
    module
        .functions
        .values()
        .find(|f| {
            f.name
                .resolve_global()
                .as_deref()
                .map(|n| n.starts_with("__lambda_"))
                .unwrap_or(false)
        })
        .expect("closure function should be present in lowered module")
}

#[test]
fn expression_bodied_closure_emits_call_to_extern() {
    let mut arena = AstArena::new();
    let body = TypedLambdaBody::Expression(Box::new(call_sink(&mut arena, 42)));
    let program = build_program(body, &mut arena);
    let module = lower(program, arena);

    let closure_fn = find_closure_fn(&module);
    let entry = closure_fn
        .blocks
        .get(&closure_fn.entry_block)
        .expect("closure entry block");

    let has_call = entry
        .instructions
        .iter()
        .any(|inst| matches!(inst, HirInstruction::Call { .. }));
    assert!(
        has_call,
        "Expression-bodied closure `def(): sink(42)` should lower to a \
         Call instruction; entry block has none. Instructions: {:?}",
        entry
            .instructions
            .iter()
            .map(std::mem::discriminant)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn block_bodied_closure_runs_each_statement() {
    let mut arena = AstArena::new();
    let call1 = call_sink(&mut arena, 1);
    let call2 = call_sink(&mut arena, 2);
    let body = TypedLambdaBody::Block(TypedBlock {
        statements: vec![
            typed_node(
                TypedStatement::Expression(Box::new(call1)),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            typed_node(
                TypedStatement::Expression(Box::new(call2)),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        span: span(),
    });
    let program = build_program(body, &mut arena);
    let module = lower(program, arena);

    let closure_fn = find_closure_fn(&module);
    let entry = closure_fn
        .blocks
        .get(&closure_fn.entry_block)
        .expect("closure entry block");

    let n_calls = entry
        .instructions
        .iter()
        .filter(|inst| matches!(inst, HirInstruction::Call { .. }))
        .count();
    assert_eq!(
        n_calls,
        2,
        "Block-bodied closure with two `sink(_)` statements should emit \
         two Call instructions; got {}. Instructions: {:?}",
        n_calls,
        entry
            .instructions
            .iter()
            .map(std::mem::discriminant)
            .collect::<Vec<_>>(),
    );
}
