//! Phase 0.3 of the fiber×effect×async plan: incoherent compositions
//! must be rejected with a clean, descriptive error instead of silently
//! miscompiling or panicking codegen.
//!
//! Each guardrail here turns a formerly-silent failure into a hard,
//! actionable compile error. Some are lifted in later phases (the
//! resumable-effect-in-fiber one becomes valid once the handler stack
//! is made fiber-aware); when that happens, flip the assertion.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

/// Compile `src` through the runtime and return the error string, or
/// `None` if it compiled cleanly.
fn compile_error(src: &str) -> Option<String> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<guardrail>")
        .expect("source should parse (the guardrail is a lowering-time check, not a parse error)");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    match rt.compile_typed_program(program) {
        Ok(_) => None,
        Err(e) => Some(format!("{e:?}")),
    }
}

/// `await` inside a `fiber def` body is rejected — a stackful fiber has
/// no state machine to park on a future. (Permanent rejection.)
#[test]
fn await_inside_fiber_body_is_rejected() {
    let src = r#"
        fiber def bad(): i64 {
            let x = await something()
            yield x
        }
    "#;
    let err = compile_error(src).expect("await-in-fiber should be a hard compile error");
    assert!(
        err.contains("await") && err.contains("fiber"),
        "error should name the await/fiber conflict; got: {err}"
    );
}

/// A `fiber def` that performs a resumable effect is rejected — the
/// fiber stack switch and the effect state machine have incompatible
/// suspension models. (Lifted in Phase 4/5 once the handler stack is
/// fiber-aware — flip this assertion then.)
#[test]
fn resumable_effect_inside_fiber_is_rejected() {
    let src = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(k: Resume<i64>): i64 {
                return k(21)
            }
        }

        @effect(E)
        fiber def producer(): i64 {
            let x = op()
            yield x
        }
    "#;
    let err = compile_error(src).expect("resumable-effect-in-fiber should be a hard compile error");
    assert!(
        err.contains("resumable") && err.contains("fiber"),
        "error should name the resumable-effect/fiber conflict; got: {err}"
    );
}

/// Regression guard: a resumable effect performed from a NON-fiber
/// function must still compile — the guardrail must not over-reject.
#[test]
fn resumable_effect_in_plain_function_still_compiles() {
    let src = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(k: Resume<i64>): i64 {
                return k(21)
            }
        }

        @effect(E)
        def run(): i64 {
            let x = op()
            return x * 2
        }

        def main() {
            let _ = run()
        }
    "#;
    assert!(
        compile_error(src).is_none(),
        "non-fiber resumable effect must still compile"
    );
}

/// The `@with(H)` annotation is removed — it never applied handler
/// scoping (silent no-op). It now fails to compile with a message
/// pointing at the `with H { }` block statement that replaces it.
#[test]
fn at_with_annotation_is_rejected() {
    let src = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(): i64 { return 5 }
        }

        @effect(E)
        @with(H)
        def run(): i64 {
            return op()
        }

        def main(): i64 {
            return run()
        }
    "#;
    let err = compile_error(src).expect("@with should be a hard compile error");
    assert!(
        err.contains("handler-scoping") && err.contains("no longer supported"),
        "error should describe the removed handler-scoping annotation in neutral \
         (non-frontend-syntax) terms; got: {err}"
    );
}

/// A resumable effect operation handled by BOTH an async and a non-async
/// handler is rejected — the perform site cannot pick the calling convention
/// statically when the handler is chosen at runtime by handler-scope
/// dispatch. (Lifted when the op-table carries a runtime async bit.)
#[test]
fn mixed_async_and_sync_handlers_rejected() {
    let src = r#"
        effect E { def op(): i64 }
        handler H1 for E { def op(k: Resume<i64>): i64 { return k(1) } }
        handler H2 for E { async def op(k: Resume<i64>): i64 { await sleep(10) return k(2) } }
        @effect(E)
        async def work(): i64 { return op() }
        async def run(): i64 { var v: i64 = 0 with H1 { v = await work() } return v }
    "#;
    let err = compile_error(src).expect("mixed async/sync handlers should be a hard error");
    assert!(
        err.contains("both async and non-async") && err.contains("op"),
        "error should name the mixed-async-ness conflict; got: {err}"
    );
}
