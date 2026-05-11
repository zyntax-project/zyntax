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
use zyntax_typed_ast::type_registry::{
    NullabilityKind, PrimitiveType, Type, TypeMetadata, Visibility,
};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::{InternedString, TypeRegistry};

/// Build a TypedProgram with a *resumable* handler:
///
///   effect State { def get(): i64 }
///   handler StateHandler for State { def get(k: Resume<i64>): i64 { return k(42) } }
///   @effect(State) def run(): i64 { return get() }
///
/// Tier 3 chain end-to-end:
///   1. SSA detects `k(42)` (k is Resume<T>-typed param) → rewrites to
///      Call(Symbol("__zyntax_effect_resume"), [k, 42]).
///   2. The runtime symbol returns 42 (placeholder pass-through).
///   3. Handler returns 42.
///   4. Cranelift's PerformEffect dispatch sees `impl_.is_resumable = true`,
///      pads handler args with a Resume sentinel.
///   5. `run()` returns 42.
fn build_resumable_effect_program() -> TypedProgram {
    // Register Resume<T> in the program's type_registry so the
    // lowering context can detect its name.
    let mut registry = TypeRegistry::new();
    let resume_type_id = registry.register_atomic_type(
        InternedString::new_global("Resume"),
        TypeMetadata::default(),
        span(),
    );
    let resume_ty = Type::Named {
        id: resume_type_id,
        type_args: vec![Type::Primitive(PrimitiveType::I64)],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // effect State { def get(): i64 }
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

    // handler StateHandler for State { def get(k: Resume<i64>): i64 { return k(42) } }
    // Body: return k(42)
    let k_var = TypedNode::new(
        TypedExpression::Variable(InternedString::new_global("k")),
        resume_ty.clone(),
        span(),
    );
    let resume_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(k_var),
            positional_args: vec![TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(42)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let handler_body = TypedNode::new(
        TypedStatement::Return(Some(Box::new(resume_call))),
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
            params: vec![TypedParameter {
                name: InternedString::new_global("k"),
                ty: resume_ty,
                ..Default::default()
            }],
            body: Some(TypedBlock {
                statements: vec![handler_body],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    // @effect(State) def run(): i64 { return get() }
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
        type_registry: registry,
        ..Default::default()
    }
}

#[test]
fn tier3_resumable_effect_executes_through_runtime() {
    // Phase H Tier 3 e2e (placeholder Resume<T> ABI): a fn calling
    // a resumable handler runs end-to-end through the JIT and
    // produces the resumed value.
    //
    // Flow:
    //   - run()'s PerformEffect(get) lowers to Call(StateHandler$get, [sentinel])
    //     (Cranelift backend pads the resumable handler's args with an i64
    //     Resume<T> sentinel)
    //   - StateHandler$get(k) body is `return k(42)`, which the SSA
    //     builder rewrote to Call(Symbol("__zyntax_effect_resume"), [k, 42])
    //   - The runtime symbol returns 42 (placeholder pass-through)
    //   - Handler returns 42, run() returns 42
    let program = build_resumable_effect_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    let exported = runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for resumable effect");
    assert!(
        exported.iter().any(|n| n == "run"),
        "run should be exported; got {:?}",
        exported
    );
    assert!(
        exported.iter().any(|n| n == "StateHandler$get"),
        "StateHandler$get (resumable) should be exported; got {:?}",
        exported
    );

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute the resumable Tier 3 effect");
    assert_eq!(
        result.as_i64(),
        Some(42),
        "Tier 3: run() → perform(get) → StateHandler.get(k) → k(42) → __zyntax_effect_resume → 42; got {:?}",
        result
    );
}

/// Build a TypedProgram with an *abort-pattern* (exception-like)
/// resumable handler:
///
///   effect E { def op(): i64 }
///   handler H for E { def op(k: Resume<i64>): i64 { return abort(-7) } }
///   @effect(E) def run(): i64 { return op() }
///
/// The handler chooses to *abort* rather than resume: `abort(-7)`
/// returns -7 through the placeholder runtime symbol. Under the
/// current placeholder ABI both `resume(v)` and `abort(v)` produce
/// the same observable effect (handler returns v, caller gets v).
/// The full Tier 3 ABI distinguishes them by unwinding the caller's
/// state machine on abort. This test verifies the abort *rewrite*
/// fires: `abort(-7)` inside a resumable handler body becomes
/// `Call(Symbol("__zyntax_effect_abort"), [-7])`.
fn build_abort_pattern_program() -> TypedProgram {
    let mut registry = TypeRegistry::new();
    let resume_type_id = registry.register_atomic_type(
        InternedString::new_global("Resume"),
        TypeMetadata::default(),
        span(),
    );
    let resume_ty = Type::Named {
        id: resume_type_id,
        type_args: vec![Type::Primitive(PrimitiveType::I64)],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::NonNull,
    };

    let e_effect = TypedEffect {
        name: InternedString::new_global("E"),
        type_params: vec![],
        operations: vec![TypedEffectOp {
            name: InternedString::new_global("op"),
            type_params: vec![],
            params: vec![],
            return_type: Type::Primitive(PrimitiveType::I64),
            span: span(),
        }],
        span: span(),
    };

    // Body: return abort(-7)
    let abort_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("abort")),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
            positional_args: vec![TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(-7)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let handler_body = TypedNode::new(
        TypedStatement::Return(Some(Box::new(abort_call))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let handler = TypedEffectHandler {
        name: InternedString::new_global("H"),
        effect_name: InternedString::new_global("E"),
        type_params: vec![],
        fields: vec![],
        handlers: vec![TypedEffectHandlerImpl {
            op_name: InternedString::new_global("op"),
            return_type: Type::Primitive(PrimitiveType::I64),
            params: vec![TypedParameter {
                name: InternedString::new_global("k"),
                ty: resume_ty,
                ..Default::default()
            }],
            body: Some(TypedBlock {
                statements: vec![handler_body],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    let op_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("op")),
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
    let return_op = TypedNode::new(
        TypedStatement::Return(Some(Box::new(op_call))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let run_fn = TypedFunction {
        name: InternedString::new_global("run"),
        return_type: Type::Primitive(PrimitiveType::I64),
        body: Some(TypedBlock {
            statements: vec![return_op],
            span: span(),
        }),
        annotations: vec![TypedAnnotation {
            name: InternedString::new_global("effect"),
            args: vec![ident_arg("E")],
            span: span(),
        }],
        visibility: Visibility::Public,
        ..Default::default()
    };

    TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Effect(e_effect),
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
        type_registry: registry,
        ..Default::default()
    }
}

#[test]
fn tier3_abort_pattern_executes_through_runtime() {
    // Phase H Tier 3 abort pattern: the handler chooses NOT to resume
    // — it calls `abort(-7)` instead, which the SSA builder rewrites
    // to `Call(Symbol("__zyntax_effect_abort"), [-7])`. The runtime
    // symbol returns -7 (placeholder ABI), the handler returns -7,
    // and run() receives -7 as the perform's result.
    let program = build_abort_pattern_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for abort-pattern handler");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute the abort-pattern handler");
    assert_eq!(
        result.as_i64(),
        Some(-7),
        "abort pattern: run() → perform(op) → H.op(k) → abort(-7) → __zyntax_effect_abort → -7; got {:?}",
        result
    );
}

/// Build a TypedProgram that exercises the *real* Resume<T> ABI:
/// the handler invokes its continuation `k(v)` and then does
/// post-resume computation, whose result is what the caller sees.
///
///   effect E { def op(): i64 }
///   handler H for E {
///       def op(k: Resume<i64>): i64 {
///           return k(21) + 1000
///       }
///   }
///   @effect(E) def run(): i64 {
///       let x = op()      // suspends here; resumes with v
///       return x * 2      // ← post-resume code; runs INSIDE k(21)
///   }
///
/// Under the placeholder ABI (Phase H): handler returns whatever
/// `__zyntax_effect_resume` passed through (21), plus 1000 = 1021.
/// Then caller's post-perform code runs with x = 1021, returning 2042.
///
/// Under the REAL ABI (Phase I.4 active): `k(21)` re-enters the
/// caller's continuation, which runs `return 21 * 2 = 42`. The
/// runtime symbol returns 42 to the handler. Handler adds 1000 →
/// returns 1042. The caller's post-perform code does NOT re-run
/// (the I.2 refinement makes the yield_block return the handler's
/// value directly), so the final result is 1042.
fn build_breakthrough_program() -> TypedProgram {
    let mut registry = TypeRegistry::new();
    let resume_type_id = registry.register_atomic_type(
        InternedString::new_global("Resume"),
        TypeMetadata::default(),
        span(),
    );
    let resume_ty = Type::Named {
        id: resume_type_id,
        type_args: vec![Type::Primitive(PrimitiveType::I64)],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // effect E { def op(): i64 }
    let e_effect = TypedEffect {
        name: InternedString::new_global("E"),
        type_params: vec![],
        operations: vec![TypedEffectOp {
            name: InternedString::new_global("op"),
            type_params: vec![],
            params: vec![],
            return_type: Type::Primitive(PrimitiveType::I64),
            span: span(),
        }],
        span: span(),
    };

    // Handler body: return k(21) + 1000
    let k_var = TypedNode::new(
        TypedExpression::Variable(InternedString::new_global("k")),
        resume_ty.clone(),
        span(),
    );
    let k_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(k_var),
            positional_args: vec![TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(21)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let k_plus_1000 = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Add,
            left: Box::new(k_call),
            right: Box::new(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(1000)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let handler_body = TypedNode::new(
        TypedStatement::Return(Some(Box::new(k_plus_1000))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let handler = TypedEffectHandler {
        name: InternedString::new_global("H"),
        effect_name: InternedString::new_global("E"),
        type_params: vec![],
        fields: vec![],
        handlers: vec![TypedEffectHandlerImpl {
            op_name: InternedString::new_global("op"),
            return_type: Type::Primitive(PrimitiveType::I64),
            params: vec![TypedParameter {
                name: InternedString::new_global("k"),
                ty: resume_ty,
                ..Default::default()
            }],
            body: Some(TypedBlock {
                statements: vec![handler_body],
                span: span(),
            }),
            ..Default::default()
        }],
        span: span(),
    };

    // run() body: let x = op(); return x * 2
    let op_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("op")),
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
    let let_x_eq_op = TypedNode::new(
        TypedStatement::Let(TypedLet {
            name: InternedString::new_global("x"),
            ty: Type::Primitive(PrimitiveType::I64),
            initializer: Some(Box::new(op_call)),
            mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
            span: span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );
    let x_var = TypedNode::new(
        TypedExpression::Variable(InternedString::new_global("x")),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let x_times_2 = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Mul,
            left: Box::new(x_var),
            right: Box::new(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(2)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let return_x2 = TypedNode::new(
        TypedStatement::Return(Some(Box::new(x_times_2))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let run_fn = TypedFunction {
        name: InternedString::new_global("run"),
        return_type: Type::Primitive(PrimitiveType::I64),
        body: Some(TypedBlock {
            statements: vec![let_x_eq_op, return_x2],
            span: span(),
        }),
        annotations: vec![TypedAnnotation {
            name: InternedString::new_global("effect"),
            args: vec![ident_arg("E")],
            span: span(),
        }],
        visibility: Visibility::Public,
        ..Default::default()
    };

    TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Effect(e_effect),
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
        type_registry: registry,
        ..Default::default()
    }
}

#[test]
fn phase_i5_breakthrough_real_resume_continuation() {
    // Phase I.5 — the breakthrough: the caller's post-perform code
    // (`return x * 2`) executes INSIDE the handler's `k(21)` call.
    // The handler adds 1000 to whatever `k(21)` returned (= 42) and
    // returns 1042. The caller's poll loop sees Ready(1042) and
    // terminates without re-running the post-perform code.
    //
    // This is the single test that demonstrably distinguishes the
    // real Resume<T> ABI from the placeholder. Under the placeholder,
    // this test would return 2042 (placeholder passes 21 through,
    // handler returns 1021, post-perform runs on 1021, returns 2042).
    let program = build_breakthrough_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for breakthrough test");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute the breakthrough test");
    assert_eq!(
        result.as_i64(),
        Some(1042),
        "BREAKTHROUGH: k(21) re-entered the caller's continuation (x * 2 = 42), \
         handler added 1000 → 1042. Got {:?} — if 2042, the post-perform code \
         ran a second time (Phase I.2 refinement regression). \
         If 1021, the runtime symbol is back to placeholder pass-through.",
        result
    );
}

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
