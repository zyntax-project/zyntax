//! Regression coverage for the lambda-body-drop bug
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
    hir::{HirCallable, HirInstruction},
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

/// Regression test for the reported issue ("status of the local checkout"
/// section): the lambda-body fix shouldn't cause OTHER top-level
/// functions to disappear from the lowered module. The bug report
/// observed `Ok([])` — empty function list — when the program had
/// both a closure-using function AND a sibling top-level fn.
///
/// Build a TypedProgram with three decls:
///   - `extern fn sink(value: i32): i32`
///   - `def helper(): i64 { return 99 }` — sibling top-level fn
///     with no lambda inside.
///   - `def main(): i64 { let f = def(): sink(42); return 0 }` —
///     same closure shape as `expression_bodied_closure_emits_call_to_extern`.
///
/// Assert: the lowered HIR module contains BOTH `helper` AND `main`
/// AND a `__lambda_*` function. Failure mode would be a missing
/// `helper` (or worse, empty `module.functions`).
#[test]
fn sibling_top_level_fns_survive_closure_lowering() {
    let mut arena = AstArena::new();

    // Sibling function with NO closure.
    let helper_name = arena.intern_string("helper");
    let mut helper_fn = TypedFunction::default();
    helper_fn.name = helper_name;
    helper_fn.return_type = Type::Primitive(PrimitiveType::I64);
    helper_fn.body = Some(TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(99)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )))),
            Type::Primitive(PrimitiveType::Unit),
            span(),
        )],
        span: span(),
    });

    // Closure-containing function (shape from
    // `expression_bodied_closure_emits_call_to_extern`).
    let closure_body = TypedLambdaBody::Expression(Box::new(call_sink(&mut arena, 42)));
    let let_stmt = let_f_eq_closure(&mut arena, closure_body);
    let main_name = arena.intern_string("main");
    let mut main_fn = TypedFunction::default();
    main_fn.name = main_name;
    main_fn.return_type = Type::Primitive(PrimitiveType::I32);
    main_fn.body = Some(TypedBlock {
        statements: vec![let_stmt, return_zero()],
        span: span(),
    });

    let mut program = TypedProgram {
        declarations: vec![
            typed_node(
                make_extern_sink(&mut arena),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            typed_node(
                TypedDeclaration::Function(helper_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            typed_node(
                TypedDeclaration::Function(main_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("closure_test");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let module = ctx
        .lower_program(&mut program)
        .expect("lower three-decl program");
    std::env::remove_var("SKIP_TYPE_CHECK");

    let names: Vec<String> = module
        .functions
        .values()
        .filter_map(|f| f.name.resolve_global())
        .collect();

    // Expect at least: sink (extern), helper, main, __lambda_*
    let has_helper = names.iter().any(|n| n == "helper");
    let has_main = names.iter().any(|n| n == "main");
    let has_lambda = names.iter().any(|n| n.starts_with("__lambda_"));

    assert!(
        has_helper && has_main && has_lambda,
        "expected the module to contain helper + main + a __lambda_* function. \
         got: {:?}",
        names
    );
}

/// Closer to the reported shape: the lambda is passed as a
/// CALL ARGUMENT (not assigned to a let), and the lambda body
/// references an outer captured variable. This is the form
/// `Div(on_click = || { count.set(count.get() + 1) })` reduces to
/// after a frontend produces TypedAst.
///
/// Surfacing this as a separate test in case `let f = lambda` and
/// `extern_call(lambda)` go through different lowering paths.
#[test]
fn lambda_as_call_arg_with_capture_survives() {
    let mut arena = AstArena::new();

    // extern fn ext_call(handler: i64): i64
    // We use i64 for the lambda-as-handler position since
    // closures lower to opaque function pointers in the i64-funneled
    // ABI.
    let ext_name = arena.intern_string("ext_call");
    let handler_name = arena.intern_string("handler");
    let mut handler_param = TypedParameter::default();
    handler_param.name = handler_name;
    handler_param.ty = Type::Primitive(PrimitiveType::I64);
    handler_param.span = span();
    let mut ext_fn = TypedFunction::default();
    ext_fn.name = ext_name;
    ext_fn.params = vec![handler_param];
    ext_fn.return_type = Type::Primitive(PrimitiveType::I64);
    ext_fn.body = None;
    ext_fn.is_external = true;
    let ext_decl = TypedDeclaration::Function(ext_fn);

    // sink — used inside the lambda body (call-with-capture)
    let sink_decl = make_extern_sink(&mut arena);

    // render_view fn:
    //   def render_view(): i64 {
    //       let count = 5
    //       return ext_call(def(): sink(count))
    //   }
    let count_name = arena.intern_string("count");
    let five = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(5)),
        Type::Primitive(PrimitiveType::I32),
        span(),
    );
    let let_count = typed_node(
        TypedStatement::Let(TypedLet {
            name: count_name,
            ty: Type::Primitive(PrimitiveType::I32),
            mutability: zyntax_typed_ast::Mutability::Immutable,
            initializer: Some(Box::new(five)),
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    // sink(count) — lambda body, captures `count`
    let sink_lookup = arena.intern_string("sink");
    let count_ref = typed_node(
        TypedExpression::Variable(count_name),
        Type::Primitive(PrimitiveType::I32),
        span(),
    );
    let sink_call = typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(
                TypedExpression::Variable(sink_lookup),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )),
            type_args: vec![],
            positional_args: vec![count_ref],
            named_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I32),
        span(),
    );

    // def(): sink(count)
    let lambda = typed_node(
        TypedExpression::Lambda(TypedLambda {
            params: vec![],
            body: TypedLambdaBody::Expression(Box::new(sink_call)),
            captures: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );

    // ext_call(<lambda>)
    let ext_lookup = arena.intern_string("ext_call");
    let ext_call_expr = typed_node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(typed_node(
                TypedExpression::Variable(ext_lookup),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )),
            type_args: vec![],
            positional_args: vec![lambda],
            named_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(ext_call_expr))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let render_view_name = arena.intern_string("render_view");
    let mut render_view_fn = TypedFunction::default();
    render_view_fn.name = render_view_name;
    render_view_fn.return_type = Type::Primitive(PrimitiveType::I64);
    render_view_fn.body = Some(TypedBlock {
        statements: vec![let_count, return_stmt],
        span: span(),
    });

    let mut program = TypedProgram {
        declarations: vec![
            typed_node(ext_decl, Type::Primitive(PrimitiveType::Unit), span()),
            typed_node(sink_decl, Type::Primitive(PrimitiveType::Unit), span()),
            typed_node(
                TypedDeclaration::Function(render_view_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("frontend_shape");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let module = ctx
        .lower_program(&mut program)
        .expect("lower frontend-shape program");
    std::env::remove_var("SKIP_TYPE_CHECK");

    let names: Vec<String> = module
        .functions
        .values()
        .filter_map(|f| f.name.resolve_global())
        .collect();
    let has_render_view = names.iter().any(|n| n == "render_view");
    let has_lambda = names.iter().any(|n| n.starts_with("__lambda_"));

    assert!(
        has_render_view,
        "render_view missing from lowered module. functions present: {:?}",
        names
    );
    assert!(
        has_lambda,
        "__lambda_* missing from lowered module. functions present: {:?}",
        names
    );

    // Regression check: inside the lambda body, the
    // call to `sink` (a known extern) should lower as
    // `HirCallable::Symbol` or `Function`, NOT `Indirect`. An
    // `Indirect` here means the lambda-body translator failed to
    // resolve the extern through `function_symbols` /
    // `extern_link_names` the same way the outer translator does,
    // which downstream surfaces as the Cranelift "function pointer
    // not in value_map" error in the caller.
    let lambda_fn = module
        .functions
        .values()
        .find(|f| {
            f.name
                .resolve_global()
                .as_deref()
                .map(|n| n.starts_with("__lambda_"))
                .unwrap_or(false)
        })
        .expect("lambda function present");
    for block in lambda_fn.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Call { callee, .. } = inst {
                if let HirCallable::Indirect(value_id) = callee {
                    panic!(
                        "Lambda body's call lowered as \
                         HirCallable::Indirect({:?}), but `sink` is \
                         a known extern function. function_symbols / \
                         extern_link_names should resolve it as Symbol \
                         the same way the outer scope does.",
                        value_id
                    );
                }
            }
        }
    }
}

/// Repro: a function that calls an extern
/// in BOTH its outer body AND inside a nested lambda. The outer
/// call should lower as `HirCallable::Symbol`/`Function`; the
/// The reported bug is that the inner (lambda-body) call
/// lowered as `HirCallable::Indirect` because the function-symbol
/// resolution path didn't fire.
///
/// If this test ever fails, the bug has reproduced — share the
/// failing call's lower output with the reporter's dump.
#[test]
fn lambda_body_extern_call_resolves_same_as_outer() {
    let mut arena = AstArena::new();
    let sink_decl = make_extern_sink(&mut arena);

    // Outer fn body:
    //   let _x = sink(10)                    ;; outer call, should be Symbol
    //   let f = def(): sink(20)              ;; inner call, ALSO should be Symbol
    //   return 0
    let outer_sink_call = call_sink(&mut arena, 10);
    let inner_sink_call = call_sink(&mut arena, 20);
    let x_name = arena.intern_string("_x");
    let let_outer_call = typed_node(
        TypedStatement::Let(TypedLet {
            name: x_name,
            ty: Type::Primitive(PrimitiveType::I32),
            mutability: zyntax_typed_ast::Mutability::Immutable,
            initializer: Some(Box::new(outer_sink_call)),
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let lambda_body = TypedLambdaBody::Expression(Box::new(inner_sink_call));
    let let_closure = let_f_eq_closure(&mut arena, lambda_body);

    let main_name = arena.intern_string("main");
    let mut main_fn = TypedFunction::default();
    main_fn.name = main_name;
    main_fn.return_type = Type::Primitive(PrimitiveType::I32);
    main_fn.body = Some(TypedBlock {
        statements: vec![let_outer_call, let_closure, return_zero()],
        span: span(),
    });

    let mut program = TypedProgram {
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
    };

    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("layer4_repro");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let module = ctx
        .lower_program(&mut program)
        .expect("lower outer + inner extern-call program");
    std::env::remove_var("SKIP_TYPE_CHECK");

    // Find both main and the lambda; check the callee variant in
    // each. If the outer resolves Symbol but inner resolves
    // Indirect, the reported regression has reproduced.
    let main_fn_hir = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some("main"))
        .expect("main present");
    let lambda_fn = module
        .functions
        .values()
        .find(|f| {
            f.name
                .resolve_global()
                .as_deref()
                .map(|n| n.starts_with("__lambda_"))
                .unwrap_or(false)
        })
        .expect("lambda present");

    fn callee_kinds(func: &zyntax_compiler::hir::HirFunction) -> Vec<&'static str> {
        let mut out = Vec::new();
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Call { callee, .. } = inst {
                    out.push(match callee {
                        HirCallable::Symbol(_) => "Symbol",
                        HirCallable::Function(_) => "Function",
                        HirCallable::Indirect(_) => "Indirect",
                        HirCallable::Intrinsic(_) => "Intrinsic",
                        HirCallable::FuncRef(_) => "FuncRef",
                    });
                }
            }
        }
        out
    }

    let outer_kinds = callee_kinds(main_fn_hir);
    let inner_kinds = callee_kinds(lambda_fn);
    assert!(
        outer_kinds
            .iter()
            .any(|k| *k == "Symbol" || *k == "Function"),
        "outer call to `sink` should resolve as Symbol/Function. Got: {:?}",
        outer_kinds,
    );
    assert!(
        !inner_kinds.iter().any(|k| *k == "Indirect"),
        "lambda body's call to `sink` lowered as Indirect, but outer \
         resolves it as Symbol/Function. This is the reported \
         regression. Outer: {:?}, Inner: {:?}",
        outer_kinds,
        inner_kinds,
    );
}

/// Reproduces the reported bug
/// "Suggested minimum to reproduce in a Zyntax-only test": a
/// non-extern `render_view` alongside ~20 sibling extern decls
/// silently drops `render_view` from the lowered HIR module.
///
/// The reporter's instrumented diagnostic showed:
///   typed_program: 20 functions including non-extern `render_view`
///   after lower_typed_program: 19 functions, all extern,
///   `render_view` missing.
///
/// This test mirrors that mix — one non-extern + 19 externs
/// covering the union of name shapes a frontend emits (`$Frontend$…`,
/// `__signal_get_…`, `__set_overlay_…`, etc.).
#[test]
fn many_externs_dont_drop_non_extern_render_view() {
    let mut arena = AstArena::new();

    // 19 externs spanning the name shapes from the bug report.
    let extern_names = [
        "__set_overlay_corner_radius__",
        "__signal_get_i32",
        "__set_overlay_border_width__",
        "text",
        "$Frontend$text",
        "__set_overlay_border_color__",
        "__signal_get_string",
        "__signal_get_f64",
        "__new_child_list__",
        "__push_child__",
        "text_int",
        "$Frontend$text_int",
        "__set_overlay_opacity__",
        "__set_overlay_bg__",
        "__new_style_overlay__",
        "__fstring_format__",
        "$Frontend$format_int",
        "string_concat",
        "$Frontend$string_concat",
    ];
    let mut decls: Vec<zyntax_typed_ast::TypedNode<TypedDeclaration>> = extern_names
        .iter()
        .map(|name| {
            let n = arena.intern_string(name);
            let mut f = TypedFunction::default();
            f.name = n;
            f.return_type = Type::Primitive(PrimitiveType::I64);
            f.body = None;
            f.is_external = true;
            typed_node(
                TypedDeclaration::Function(f),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )
        })
        .collect();

    // Non-extern `render_view`. Body just returns 0 — we're testing
    // whether it SURVIVES lowering, not what it computes.
    let render_view_name = arena.intern_string("render_view");
    let mut render_view_fn = TypedFunction::default();
    render_view_fn.name = render_view_name;
    render_view_fn.return_type = Type::Primitive(PrimitiveType::I64);
    render_view_fn.body = Some(TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Return(Some(Box::new(typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(0)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )))),
            Type::Primitive(PrimitiveType::Unit),
            span(),
        )],
        span: span(),
    });
    decls.push(typed_node(
        TypedDeclaration::Function(render_view_fn),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    ));

    let mut program = TypedProgram {
        declarations: decls,
        span: span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("frontend_repro");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let module = ctx
        .lower_program(&mut program)
        .expect("lower 20-decl frontend-shape program");
    std::env::remove_var("SKIP_TYPE_CHECK");

    let non_extern_names: Vec<String> = module
        .functions
        .values()
        .filter(|f| !f.is_external)
        .filter_map(|f| f.name.resolve_global())
        .collect();
    assert!(
        non_extern_names.iter().any(|n| n == "render_view"),
        "render_view dropped from lowered module — repro of the \
         reported bug. Non-extern functions present: {:?}. Total \
         functions: {}.",
        non_extern_names,
        module.functions.len(),
    );
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

    // Stronger assertion: the call to
    // `sink` should lower as `HirCallable::Symbol` or
    // `HirCallable::Function`, NOT `Indirect`. The latter would
    // mean the lambda-body translator failed to resolve `sink`
    // through `function_symbols` / `extern_link_names` the same
    // way the outer translator does — leading to the Cranelift
    // "function pointer not in value_map" error in the caller.
    for inst in &entry.instructions {
        if let HirInstruction::Call { callee, .. } = inst {
            match callee {
                HirCallable::Symbol(_) | HirCallable::Function(_) => {
                    // Direct call — what we expect.
                }
                HirCallable::Indirect(_) => {
                    panic!(
                        "Lambda body's call to `sink` lowered as \
                         HirCallable::Indirect, but `sink` is a known \
                         extern function and the outer translator \
                         resolves it as Symbol. This is the reported \
                         layer-4 regression: function-name resolution \
                         doesn't work inside lambda bodies."
                    );
                }
                other => panic!("unexpected callee variant: {:?}", other),
            }
        }
    }
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

/// Regression: when a lambda body
/// calls a function name that resolves to NOTHING in scope —
/// undeclared as extern, undefined as a local — the SSA Call lowering
/// must surface a clean `CompilerError::Lowering` rather than fall
/// through to `HirCallable::Indirect(Undef)`. The Indirect-of-Undef
/// path was the reported "+1 lambda fails verification, Reset lambda
/// SIGSEGVs" symmetric pair: an indirect call through a null pointer
/// either trips Cranelift's verifier (best case) or JITs to address 0
/// (worst case). Both hide the real bug — the embedder forgot to
/// declare a host-provided extern. This test asserts the embedder gets
/// the error instead of a crash.
#[test]
fn undefined_callee_in_lambda_body_surfaces_lowering_error() {
    let mut arena = AstArena::new();

    // Build a Call to a name we never declare:
    //     undefined_extern()
    // not registered as extern, not in scope.
    let undefined_name = arena.intern_string("undefined_extern");
    let undefined_call = typed_node(
        TypedExpression::Call(zyntax_typed_ast::TypedCall {
            callee: Box::new(typed_node(
                TypedExpression::Variable(undefined_name),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            )),
            type_args: vec![],
            positional_args: vec![],
            named_args: vec![],
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    // Wrap in a Block-bodied lambda: `|| { undefined_extern() }`.
    let body = TypedLambdaBody::Block(TypedBlock {
        statements: vec![typed_node(
            TypedStatement::Expression(Box::new(undefined_call)),
            Type::Primitive(PrimitiveType::Unit),
            span(),
        )],
        span: span(),
    });

    let program = build_program(body, &mut arena);

    // Run lowering manually (not via the `lower()` helper, which panics
    // on Err) so we can assert on the error shape.
    std::env::set_var("SKIP_TYPE_CHECK", "1");
    let type_registry = Arc::new(TypeRegistry::new());
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("undefined_callee_test");
    let arena = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry, arena, config);
    let result = {
        let mut prog = program;
        ctx.lower_program(&mut prog)
    };
    std::env::remove_var("SKIP_TYPE_CHECK");

    let err = result.expect_err(
        "Lowering a lambda body that calls an undeclared function must \
         return Err — falling through to Indirect(Undef) hides the \
         missing-extern bug as a runtime SIGSEGV.",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("undefined_extern") && msg.to_lowercase().contains("undefined"),
        "Expected a clear 'call to undefined function undefined_extern' \
         error message, got: {}",
        msg
    );
}
