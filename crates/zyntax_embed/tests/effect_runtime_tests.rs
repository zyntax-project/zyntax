//! Phase H, M6 — End-to-end Tier 1 algebraic effects: TypedAST → HIR
//! → JIT → execute, verifying the handler dispatch lands at the
//! correct mangled function and produces the expected return value.
//!
//! These tests construct a TypedProgram directly (skipping the
//! grammar/parser layer — same approach the
//! `effect_emission_tests.rs` use). What they add over those tests:
//! they execute the compiled module through `ZyntaxRuntime::call_function`
//! and assert on the runtime return value.
//!
//! Tests cover:
//!   * `test_simple_effect_returns_handler_value` — an `@effect(State)
//!     fn run(): i64 { return get() }` whose handler returns 42; runtime
//!     output must be 42.
//!   * `test_effect_with_args_passes_through` — effect op `add(a, b)`
//!     whose handler computes a + b; runtime output must reflect the
//!     argument forwarding.

use zyntax_embed::{NativeSignature, NativeType, ZyntaxRuntime};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

#[test]
fn effect_runtime_symbols_are_registered_at_runtime_construction() {
    // Phase H Tier 3 foundation: `ZyntaxRuntime::new()` automatically
    // wires the 5 `__zyntax_effect_*` runtime symbols via
    // `register_effect_runtime_symbols`. Verify each appears in the
    // runtime's `plugin_signatures` map with the expected param count
    // (the call-site lowering reads param_count to size args).
    let runtime = ZyntaxRuntime::new().expect("runtime construction must succeed");
    let sigs = runtime.plugin_signatures();

    for (name, expected_params) in [
        ("__zyntax_effect_push_handler", 3),
        ("__zyntax_effect_pop_handler", 1),
        ("__zyntax_effect_lookup_handler", 1),
        ("__zyntax_effect_resume", 2),
        ("__zyntax_effect_abort", 1),
    ] {
        let sig = sigs
            .get(name)
            .unwrap_or_else(|| panic!("runtime symbol {name} should be registered"));
        assert_eq!(
            sig.param_count, expected_params,
            "{name}: param_count mismatch",
        );
    }
}

fn span() -> Span {
    Span::new(0, 0)
}

fn ident_arg(name: &str) -> TypedAnnotationArg {
    TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
        InternedString::new_global(name),
    ))
}

/// Build a TypedProgram with shape:
///
///   effect State { def get(): i64 }
///   handler StateHandler for State { def get(): i64 { return 42 } }
///   @effect(State) def run(): i64 { return get() }
fn build_simple_effect_program() -> TypedProgram {
    let state_effect = TypedEffect {
        name: InternedString::new_global("State"),
        type_params: vec![],
        operations: vec![TypedEffectOp {
            name: InternedString::new_global("get"),
            type_params: vec![],
            params: vec![],
            return_type: Type::Primitive(PrimitiveType::I64),
            span: span(),
        }],
        span: span(),
    };

    // The handler returns the literal 42 from `get()`.
    let return_42 = TypedNode::new(
        TypedStatement::Return(Some(Box::new(TypedNode::new(
            TypedExpression::Literal(TypedLiteral::Integer(42)),
            Type::Primitive(PrimitiveType::I64),
            span(),
        )))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let handler = TypedEffectHandler {
        name: InternedString::new_global("StateHandler"),
        effect_name: InternedString::new_global("State"),
        type_params: vec![],
        fields: vec![],
        handlers: vec![TypedEffectHandlerImpl {
            op_name: InternedString::new_global("get"),
            return_type: Type::Primitive(PrimitiveType::I64),
            params: vec![],
            body: Some(TypedBlock {
                statements: vec![return_42],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    // Body: return get()
    let get_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("get")),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
            positional_args: vec![],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let return_get = TypedNode::new(
        TypedStatement::Return(Some(Box::new(get_call))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let run_fn = TypedFunction {
        name: InternedString::new_global("run"),
        return_type: Type::Primitive(PrimitiveType::I64),
        body: Some(TypedBlock {
            statements: vec![return_get],
            span: span(),
        }),
        annotations: vec![TypedAnnotation {
            name: InternedString::new_global("effect"),
            args: vec![ident_arg("State")],
            span: span(),
        }],
        visibility: Visibility::Public,
        ..Default::default()
    };

    TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Effect(state_effect),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::EffectHandler(handler),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::Function(run_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        ..Default::default()
    }
}

#[test]
fn m6_typed_program_with_effect_compiles_via_runtime() {
    // M6 (compile-only check): the entire pipeline (TypedAST → HIR →
    // Cranelift) accepts an `@effect`-annotated function whose body
    // performs an effect operation. After M5's mangle reconciliation,
    // the PerformEffect at run() lowers to a direct call to
    // `StateHandler$get` — which exists in the same module thanks to
    // `algebraic_effects_pass::dispatch::handler_decl_to_impl`.
    let program = build_simple_effect_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    let function_names = runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for an @effect-annotated fn");
    assert!(
        function_names.iter().any(|n| n == "run"),
        "the @effect(State) fn run() must appear in the exported names; got {:?}",
        function_names
    );
    assert!(
        function_names.iter().any(|n| n == "StateHandler$get"),
        "the handler StateHandler.get must be lowered to standalone fn StateHandler$get; got {:?}",
        function_names
    );
}

/// Build a program with shape:
///
///   effect Math { def add(a: i64, b: i64): i64 }
///   handler Adder for Math { def add(a, b): i64 { return a + b } }
///   @effect(Math) def compute(): i64 { return add(10, 32) }
fn build_effect_with_args_program() -> TypedProgram {
    let math_effect = TypedEffect {
        name: InternedString::new_global("Math"),
        type_params: vec![],
        operations: vec![TypedEffectOp {
            name: InternedString::new_global("add"),
            type_params: vec![],
            params: vec![
                TypedParameter {
                    name: InternedString::new_global("a"),
                    ty: Type::Primitive(PrimitiveType::I64),
                    ..Default::default()
                },
                TypedParameter {
                    name: InternedString::new_global("b"),
                    ty: Type::Primitive(PrimitiveType::I64),
                    ..Default::default()
                },
            ],
            return_type: Type::Primitive(PrimitiveType::I64),
            span: span(),
        }],
        span: span(),
    };

    // Handler body: return a + b
    let a_var = TypedNode::new(
        TypedExpression::Variable(InternedString::new_global("a")),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let b_var = TypedNode::new(
        TypedExpression::Variable(InternedString::new_global("b")),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let a_plus_b = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Add,
            left: Box::new(a_var),
            right: Box::new(b_var),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let return_sum = TypedNode::new(
        TypedStatement::Return(Some(Box::new(a_plus_b))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let handler = TypedEffectHandler {
        name: InternedString::new_global("Adder"),
        effect_name: InternedString::new_global("Math"),
        type_params: vec![],
        fields: vec![],
        handlers: vec![TypedEffectHandlerImpl {
            op_name: InternedString::new_global("add"),
            return_type: Type::Primitive(PrimitiveType::I64),
            params: vec![
                TypedParameter {
                    name: InternedString::new_global("a"),
                    ty: Type::Primitive(PrimitiveType::I64),
                    ..Default::default()
                },
                TypedParameter {
                    name: InternedString::new_global("b"),
                    ty: Type::Primitive(PrimitiveType::I64),
                    ..Default::default()
                },
            ],
            body: Some(TypedBlock {
                statements: vec![return_sum],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    // Body: return add(10, 32)
    let add_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("add")),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
            positional_args: vec![
                TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::Integer(10)),
                    Type::Primitive(PrimitiveType::I64),
                    span(),
                ),
                TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::Integer(32)),
                    Type::Primitive(PrimitiveType::I64),
                    span(),
                ),
            ],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let return_add = TypedNode::new(
        TypedStatement::Return(Some(Box::new(add_call))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let compute_fn = TypedFunction {
        name: InternedString::new_global("compute"),
        return_type: Type::Primitive(PrimitiveType::I64),
        body: Some(TypedBlock {
            statements: vec![return_add],
            span: span(),
        }),
        annotations: vec![TypedAnnotation {
            name: InternedString::new_global("effect"),
            args: vec![ident_arg("Math")],
            span: span(),
        }],
        visibility: Visibility::Public,
        ..Default::default()
    };

    TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Effect(math_effect),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::EffectHandler(handler),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
            TypedNode::new(
                TypedDeclaration::Function(compute_fn),
                Type::Primitive(PrimitiveType::Unit),
                span(),
            ),
        ],
        ..Default::default()
    }
}

#[test]
fn m6_tier1_effect_with_args_executes_through_runtime() {
    // H-followup: Tier 1 effect with operation arguments. The handler
    // computes a + b; calling compute() with the inline args 10 + 32
    // must return 42 — proves argument forwarding through PerformEffect
    // → direct dispatch → handler call.
    let program = build_effect_with_args_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("compute", &[], &sig)
        .expect("runtime.call_function(\"compute\") should execute");
    assert_eq!(
        result.as_i64(),
        Some(42),
        "compute() → perform(add(10, 32)) → Adder.add(10, 32) → 42; got {:?}",
        result
    );
}

#[test]
fn m6_tier1_effect_executes_through_runtime() {
    // H-followup: after narrowing the krio filter to resumable-only
    // (orchestrator.rs::function_has_resumable_effect), a Tier 1
    // `@effect(State)` fn whose handler is non-resumable should keep
    // its plain `(): i64` ABI — so `runtime.call_function` can drive
    // it directly. The PerformEffect inside run() is dispatched by
    // the Cranelift backend at runtime to `StateHandler$get`, which
    // returns 42. Therefore `run()` must return 42.
    let program = build_simple_effect_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute the Tier 1 effect");
    assert_eq!(
        result.as_i64(),
        Some(42),
        "run() perform(get()) → StateHandler.get() → 42; got {:?}",
        result
    );
}
