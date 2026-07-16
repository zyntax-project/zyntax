//! Non-resumable algebraic effects — a handler op that does its work and
//! returns without capturing the continuation (`k`). These go through the
//! Cranelift static-dispatch path (not the krio resumable state machine).
//!
//! Regression test for the reachability-DCE gap: `PerformEffect` didn't
//! mark its handler op fns reachable, so their bodies were skipped in the
//! compile pass and the perform-site's direct call resolved to an
//! undefined symbol at JIT finalize.

use zynml::ZynML;

#[test]
fn non_resumable_effect_resolves_and_runs() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log {
            def emit(): i64
        }

        handler ConsoleLog for Log {
            def emit(): i64 {
                return 42
            }
        }

        @effect(Log)
        def do_work(): i64 {
            let x = emit()
            return x + 1
        }

        def main(): i64 {
            return do_work()
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(result, 43, "emit() returns 42; do_work adds 1");
}
