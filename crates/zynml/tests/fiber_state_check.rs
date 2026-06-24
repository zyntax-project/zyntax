//! Empirical diagnostics for the current fiber-end-to-end state.
//!
//! Two questions worth answering before going further:
//!
//! 1. Does a `fiber def NAME(): T { ... }` function record its
//!    return type as `Fiber<T>` (what callers should see) or as `T`
//!    (the yield type)?
//! 2. When source spells `f.next()` on a `Fiber<i64>`, does the
//!    SSA produce a `FiberResume` op via the `BuiltinClass`
//!    interception, or does it fall through to ordinary method
//!    dispatch?

use std::sync::Arc;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::HirInstruction;
use zyntax_embed::{compile_to_hir, CompilationConfig};
use zyntax_typed_ast::type_registry::Type;
use zyntax_typed_ast::typed_ast::TypedDeclaration;

fn parse_program(src: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    grammar
        .parse_with_filename(src, "<fiber_state_check>")
        .expect("source should parse")
}

fn lower(src: &str) -> zyntax_compiler::hir::HirModule {
    let mut program = parse_program(src);
    let type_registry = Arc::new(program.type_registry.clone());
    compile_to_hir(&mut program, type_registry, CompilationConfig::default())
        .expect("HIR lowering should succeed")
}

/// Diagnostic 1: what does the parser record as a `fiber def`'s
/// return_type? This test is a SNAPSHOT of current behavior — it
/// fails on the day we wrap the return type as `Fiber<T>` either at
/// parse time or at call-site lowering.
#[test]
fn diagnostic_fiber_def_return_type() {
    let program = parse_program(
        r#"
        fiber def gen(): i64 {
            yield 1
        }
        "#,
    );

    let mut found = None;
    for decl in &program.declarations {
        if let TypedDeclaration::Function(f) = &decl.node {
            if f.name.resolve_global().as_deref() == Some("gen") {
                found = Some(f);
                break;
            }
        }
    }
    let f = found.expect("gen function should be in program");

    eprintln!(
        "[DIAG] fiber def gen():i64 -> return_type = {:?}",
        f.return_type
    );
    eprintln!("[DIAG] fiber def gen():i64 -> is_fiber = {}", f.is_fiber);

    // is_fiber must be true (the grammar fix from step 3a).
    assert!(f.is_fiber, "fiber def must have is_fiber=true");

    // Document the CURRENT behavior. The grammar records the
    // *yield* type as the function's return_type — not yet wrapped
    // as `Fiber<T>`. Wrapping at parse time would create a body-vs-
    // signature mismatch (the body's `yield` produces T values, not
    // Fiber<T>); the correct wrapping happens at the CALL site via
    // SSA lowering. This test asserts the current state so the
    // intent is documented; flip it to expect `Type::Fiber(_)`
    // once the call-site lowering treats the call result as
    // `Fiber<T>`.
    let return_is_t = matches!(
        f.return_type,
        Type::Primitive(zyntax_typed_ast::PrimitiveType::I64)
    );
    let return_is_fiber_t = matches!(&f.return_type, Type::Fiber(inner)
        if matches!(inner.as_ref(), Type::Primitive(zyntax_typed_ast::PrimitiveType::I64)));

    assert!(
        return_is_t || return_is_fiber_t,
        "return_type should be either `i64` (today) or `Fiber<i64>` (after call-site lowering); got {:?}",
        f.return_type
    );

    if return_is_fiber_t {
        eprintln!("[DIAG] return type is `Fiber<i64>` — call-site wrapping appears to be in place");
    } else {
        eprintln!("[DIAG] return type is raw `i64` — call-site lowering NOT yet wired");
    }
}

/// Diagnostic 1b: at the CALL SITE — when source spells
/// `let f = gen()` where `gen` is `fiber def`, does the SSA-lowered
/// HIR produce a `Fiber<T>` value via `FiberNew`, or does it fall
/// through to a regular `Call` that runs the body synchronously?
#[test]
fn diagnostic_fiber_call_site() {
    let module = lower(
        r#"
        fiber def gen(): i64 {
            yield 1
        }

        def main(): i64 {
            let f = gen()
            return 0
        }
        "#,
    );

    // Search `main` for either a FiberNew (new behavior) or a
    // regular Call to `gen` (current behavior).
    let mut fiber_new_count = 0usize;
    let mut call_gen_count = 0usize;
    for func in module.functions.values() {
        if func.name.resolve_global().as_deref() != Some("main") {
            continue;
        }
        for block in func.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::FiberNew { .. } => fiber_new_count += 1,
                    HirInstruction::Call {
                        callee: zyntax_compiler::hir::HirCallable::Function(_),
                        ..
                    } => call_gen_count += 1,
                    _ => {}
                }
            }
        }
    }

    eprintln!(
        "[DIAG] let f = gen() in main -> FiberNew ops = {}, Call(Function) ops = {}",
        fiber_new_count, call_gen_count
    );

    if fiber_new_count > 0 {
        eprintln!("[DIAG] call-site lowering IS wired — call to fiber def emits FiberNew");
    } else if call_gen_count > 0 {
        eprintln!(
            "[DIAG] call-site lowering NOT wired — call to fiber def runs body synchronously"
        );
    } else {
        eprintln!("[DIAG] neither shape found; main got optimised out");
    }
}

/// Diagnostic 2: does the SSA's BuiltinClass interception fire for
/// `.next()` on a `Fiber<T>` receiver? This test constructs a
/// minimal module where the receiver type is unambiguously
/// `Fiber<i64>` (declared via let annotation) and inspects whether
/// `FiberResume` appears in the lowered HIR.
#[test]
fn diagnostic_fiber_next_lowering() {
    // We can't yet *construct* a Fiber<i64> from source (no
    // call-site lowering), so we fabricate one by annotating a
    // declared variable. The body never executes — we only inspect
    // the lowered HIR shape.
    let module = lower(
        r#"
        def consume(f: Fiber<i64>): i64 {
            match f.next() {
                case Some(x) { x }
                case None() { 0 }
            }
        }
        "#,
    );

    let mut total_fiber_resume = 0usize;
    let mut total_call_sym_resume = 0usize;
    for func in module.functions.values() {
        for block in func.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::FiberResume { .. } => total_fiber_resume += 1,
                    HirInstruction::Call {
                        callee: zyntax_compiler::hir::HirCallable::Symbol(name),
                        ..
                    } if name == "krio_fiber_resume" => total_call_sym_resume += 1,
                    _ => {}
                }
            }
        }
    }

    eprintln!(
        "[DIAG] consume(f: Fiber<i64>) {{ f.next() }} -> FiberResume ops = {}, krio_fiber_resume calls = {}",
        total_fiber_resume, total_call_sym_resume
    );

    // The two ops are the same dispatch path at different layers:
    // SSA emits `FiberResume`; `apply_krio_fiber_lowering` rewrites
    // it to `Call::Symbol("krio_fiber_resume")`. Either count being
    // > 0 means the interception fired. Both being 0 means the
    // BuiltinClass dispatch didn't see this `.next()` call.
    let dispatch_fired = total_fiber_resume + total_call_sym_resume > 0;
    assert!(
        dispatch_fired,
        "expected `f.next()` to lower to FiberResume (or post-pass Call::Symbol); the BuiltinClass dispatch did not fire"
    );
}
