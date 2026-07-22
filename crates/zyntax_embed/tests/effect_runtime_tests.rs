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

/// Phase J.1 candidate: multi-shot resumption. Handler calls `k(v)`
/// three times with different values and sums the results. If the
/// caller's continuation re-enters cleanly each time, the result is
/// `k(10) + k(20) + k(30) = 20 + 40 + 60 = 120`. If state leaks
/// between resumes, the result diverges.
///
///   effect E { def op(): i64 }
///   handler H for E {
///       def op(k: Resume<i64>): i64 {
///           return k(10) + k(20) + k(30)
///       }
///   }
///   @effect(E) def run(): i64 {
///       return op() * 2     // caller continuation: multiply by 2
///   }
///
/// Multi-shot in the simplest case (no captures-lift slots mutated
/// by post-perform code) works under Phase I without explicit state
/// machine cloning — each `__zyntax_effect_resume` call resets the
/// result_slot + state_slot, re-polls the same state machine, and
/// returns the result. Whether this generalises to more complex
/// post-perform code with mutable captures is Phase J.1's larger
/// question; this test is the lower-bound check.
fn build_multi_shot_program() -> TypedProgram {
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

    // Build `k(N)` expression as a call to the k param with N.
    let make_k_call = |n: i128| -> TypedNode<TypedExpression> {
        let k_var = TypedNode::new(
            TypedExpression::Variable(InternedString::new_global("k")),
            resume_ty.clone(),
            span(),
        );
        TypedNode::new(
            TypedExpression::Call(TypedCall {
                callee: Box::new(k_var),
                positional_args: vec![TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::Integer(n)),
                    Type::Primitive(PrimitiveType::I64),
                    span(),
                )],
                named_args: vec![],
                type_args: vec![],
            }),
            Type::Primitive(PrimitiveType::I64),
            span(),
        )
    };

    // Handler body: return k(10) + k(20) + k(30)
    let k10 = make_k_call(10);
    let k20 = make_k_call(20);
    let k30 = make_k_call(30);
    let k10_plus_k20 = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Add,
            left: Box::new(k10),
            right: Box::new(k20),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let sum_all = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Add,
            left: Box::new(k10_plus_k20),
            right: Box::new(k30),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let handler_body = TypedNode::new(
        TypedStatement::Return(Some(Box::new(sum_all))),
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

    // run() body: return op() * 2
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
    let op_times_2 = TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: zyntax_typed_ast::typed_ast::BinaryOp::Mul,
            left: Box::new(op_call),
            right: Box::new(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(2)),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let return_op2 = TypedNode::new(
        TypedStatement::Return(Some(Box::new(op_times_2))),
        Type::Primitive(PrimitiveType::Unit),
        span(),
    );

    let run_fn = TypedFunction {
        name: InternedString::new_global("run"),
        return_type: Type::Primitive(PrimitiveType::I64),
        body: Some(TypedBlock {
            statements: vec![return_op2],
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
fn phase_j1_multi_shot_resumption_simple_case() {
    // Multi-shot resume: handler calls k(10), k(20), k(30) and sums.
    // Each k(N) re-enters the caller's continuation `op() * 2`, so:
    //   k(10) = 20, k(20) = 40, k(30) = 60
    //   sum = 120
    // The handler returns 120 to run(), which returns 120.
    //
    // This test exists as a lower-bound check: in the no-captures-lift
    // case Phase I's single-shot path generalises trivially because
    // each `__zyntax_effect_resume` call independently sets
    // result_slot + state_slot and re-polls. If/when post-perform code
    // mutates captures-lift state across resumes, state machine
    // cloning becomes necessary (the Phase J.1 work proper).
    let program = build_multi_shot_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for multi-shot");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute multi-shot");
    assert_eq!(
        result.as_i64(),
        Some(120),
        "Multi-shot: handler k(10)+k(20)+k(30) = 20+40+60 = 120. \
         Got {:?}. If 60 (or any single-resume value), only the last k(N) \
         actually ran. If something else, state leaked between resumes.",
        result
    );
}

/// Phase J.2: build an abort-pattern program that has POST-PERFORM
/// code. The handler aborts; that code MUST NOT run.
///
///   effect E { def op(): i64 }
///   handler H for E {
///       def op(k: Resume<i64>): i64 {
///           return abort(99)        // skip caller's continuation
///       }
///   }
///   @effect(E) def run(): i64 {
///       let x = op()
///       return x * 2                // ← MUST NOT run on abort
///   }
///
/// Expected: `run() = 99` (handler returned 99 directly).
/// If abort got treated like resume (placeholder bug), the
/// post-perform `x * 2` would run with x=99 → 198.
fn build_abort_with_post_perform_program() -> TypedProgram {
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

    // Handler body: return abort(99)
    let abort_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("abort")),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
            positional_args: vec![TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(99)),
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
fn phase_j2_abort_skips_post_perform_code() {
    // Distinguishing abort from resume requires showing that the
    // caller's post-perform code DOESN'T run on abort. Handler returns
    // 99 via `abort(99)`; the caller has `let x = op(); return x * 2`.
    // With correct abort semantics: run() returns 99 (post-perform
    // skipped). With abort-treated-as-resume: 198 (post-perform ran
    // with x=99).
    //
    // The Phase I.2 refinement (yield_block returns the handler's
    // value directly instead of bumping state and re-polling) gives
    // this for free: the handler returns 99 → yield_block returns 99
    // → entry wrapper's poll loop sees Ready(99) → run() returns 99.
    let program = build_abort_with_post_perform_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");
    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for abort-with-post-perform");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute the abort test");
    assert_eq!(
        result.as_i64(),
        Some(99),
        "abort(99): handler should return 99 directly, post-perform `x * 2` MUST NOT run. \
         Got {:?}. If 198, abort was treated like resume and post-perform ran with x=99.",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────
// Phase J.3: async-out-of-line resume
// ─────────────────────────────────────────────────────────────────────

/// Test-side stash for the Resume<T> pointer. The handler calls
/// `__test_stash_resume(k)` which writes `k` here; the test then
/// invokes `__zyntax_effect_resume` directly with the stashed
/// pointer to drive the continuation out-of-line.
///
/// Concurrency: the test is single-threaded so a `static mut`
/// would be technically sound, but we use `std::sync::Mutex` to
/// keep the API tidy and ready for future multi-threaded tests.
static STASHED_RESUME: std::sync::OnceLock<std::sync::Mutex<usize>> = std::sync::OnceLock::new();

fn stashed_resume_slot() -> &'static std::sync::Mutex<usize> {
    STASHED_RESUME.get_or_init(|| std::sync::Mutex::new(0))
}

/// Sentinel value the stash routine returns to the @effect fn,
/// indicating "suspension stashed externally — return this and let
/// the host drive the resume later". Non-zero because the entry
/// wrapper's poll loop uses 0 as the Pending sentinel.
const STASH_SENTINEL: i64 = -1;

/// Handler-callable stash routine. Compiled ZynML invokes this via
/// the `$Test$stash_resume` symbol; we register it on the runtime
/// before compilation. Returns `STASH_SENTINEL` (non-zero) so the
/// poll loop breaks on Ready — the entry wrapper treats 0 as
/// Pending and would otherwise spin forever.
#[no_mangle]
extern "C" fn test_stash_resume(k: *mut u8) -> i64 {
    // Retain the state machine before stashing. `generate_sync_entry`
    // auto-releases the SM on its return path; without an explicit
    // retain here the SM would be freed before we got a chance to
    // drive the continuation out-of-line, and the stashed Resume
    // pointer would dangle. The retain bumps the refcount from 1
    // (set by sync_entry) to 2 — sync_entry's release brings it
    // back to 1, the SM stays alive for the out-of-line resume, and
    // the test's final release (or implicit leak at process exit)
    // brings it to 0 to free.
    unsafe {
        zyntax_embed::__zyntax_runtime_retain_sm(k);
    }
    let mut slot = stashed_resume_slot().lock().unwrap();
    *slot = k as usize;
    STASH_SENTINEL
}

/// Phase J.3 program shape:
///
///   effect E { def op(): i64 }
///   handler H for E {
///       def op(k: Resume<i64>): i64 {
///           return $Test$stash_resume(k)   // stash + return 0
///       }
///   }
///   @effect(E) def run(): i64 {
///       let x = op()
///       return x * 2
///   }
///
/// Synchronous run() returns 0 (handler returned 0). The Resume<T>
/// pointer is now in STASHED_RESUME. The test then invokes
/// __zyntax_effect_resume(stashed, 100) directly from Rust, which
/// re-polls the @effect fn's state machine with 100 as the perform
/// result. The post-perform code (`x * 2`) runs → returns 200. The
/// runtime symbol's return value is 200.
fn build_async_out_of_line_program() -> TypedProgram {
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

    // Handler body: return $Test$stash_resume(k)
    // The `$`-prefixed name short-circuits the SSA Call handler to
    // `HirCallable::Symbol` directly (see ssa.rs ~line 3270).
    let stash_call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("$Test$stash_resume")),
                Type::Primitive(PrimitiveType::I64),
                span(),
            )),
            positional_args: vec![TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("k")),
                resume_ty.clone(),
                span(),
            )],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::I64),
        span(),
    );
    let handler_body = TypedNode::new(
        TypedStatement::Return(Some(Box::new(stash_call))),
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
fn phase_j3_async_out_of_line_resume() {
    use zyntax_compiler::zrtl::{
        PrimitiveSize, TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig, MAX_PARAMS,
    };

    // Clear any leftover stash from a previous test invocation.
    *stashed_resume_slot().lock().unwrap() = 0;

    let program = build_async_out_of_line_program();
    let mut runtime = ZyntaxRuntime::new().expect("runtime must construct");

    // Register the test stash fn. Signature: (ptr) -> i64.
    let mut params: [TypeTag; MAX_PARAMS] = [TypeTag::VOID; MAX_PARAMS];
    params[0] = TypeTag::new(TypeCategory::Pointer, 0, TypeFlags::NONE);
    runtime.register_function_typed(
        "$Test$stash_resume",
        test_stash_resume as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::new(
                TypeCategory::Int,
                PrimitiveSize::Bits64 as u16,
                TypeFlags::NONE,
            ),
            params,
        },
    );
    // Cranelift's JITModule was constructed at runtime-creation time —
    // bind the freshly-registered symbol into it before any compile
    // call. Without this, the JIT can't resolve $Test$stash_resume at
    // link time.
    runtime
        .finalize_runtime_symbols()
        .expect("finalize_runtime_symbols should succeed");

    runtime
        .compile_typed_program(program)
        .expect("compile_typed_program must succeed for async-out-of-line");

    // First call: run() invokes handler, which stashes k and returns
    // the STASH_SENTINEL (non-zero — 0 is the poll-loop's Pending
    // sentinel so a 0 return would hang the loop).
    let sig = NativeSignature::new(&[], NativeType::I64);
    let result = runtime
        .call_function("run", &[], &sig)
        .expect("runtime.call_function(\"run\") should execute synchronously");
    assert_eq!(
        result.as_i64(),
        Some(STASH_SENTINEL),
        "Handler stashed k and returned the STASH_SENTINEL; run() propagates it. Got {:?}",
        result
    );

    // The stash slot must now hold a non-null Resume<T> pointer.
    let stashed = *stashed_resume_slot().lock().unwrap();
    assert_ne!(
        stashed, 0,
        "Handler should have stashed a non-null Resume pointer"
    );

    // Drive the continuation out-of-line: invoke __zyntax_effect_resume
    // directly with value=100. The runtime symbol re-polls the @effect
    // fn's state machine starting at next_state (resume_entry), which
    // runs `let x = 100; return x * 2` → 200.
    // The runtime symbol is `extern "C"` and the stashed pointer
    // originated from a still-live state machine — no actual unsafety
    // beyond what `extern "C"` already exposes.
    let out_of_line_result = zyntax_embed::__zyntax_effect_resume(stashed as *mut u8, 100);
    assert_eq!(
        out_of_line_result, 200,
        "Out-of-line resume with value=100 should run post-perform `x * 2` → 200. Got {}",
        out_of_line_result
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
        ("__zyntax_effect_push_handler", 4),
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
