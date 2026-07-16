//! Regional dispatch over a RESUMABLE effect — `with H { }` overrides
//! the static handler even for handlers that capture the continuation
//! (`k(v)`), which go through the krio captures-lift state machine
//! rather than the Cranelift static-dispatch path.
//!
//! The resumable perform-site now resolves its op function the same
//! way the non-resumable path does: `select(lookup_op != 0, dyn,
//! static)` + an indirect call, so a `with H` on the handler stack
//! wins and otherwise the compile-time handler is used unchanged.

use zynml::ZynML;

#[test]
fn with_block_overrides_resumable_handler() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect E {
            def op(): i64
        }

        handler H1 for E {
            def op(k: Resume<i64>): i64 { return k(1) }
        }

        handler H2 for E {
            def op(k: Resume<i64>): i64 { return k(2) }
        }

        @effect(E)
        def work(): i64 {
            let x = op()
            return x
        }

        def main(): i64 {
            let outside = work()      // static handler
            let mut inside: i64 = 0
            with H1 {
                inside = work()       // with-scoped -> H1 -> k(1) -> 1
            }
            return outside * 10 + inside
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    // `outside` uses whichever handler is the static default; `inside`
    // is forced to H1 by the `with` block (k(1) -> 1). We assert the
    // low digit is 1 (the override) — that's the regional behaviour.
    assert_eq!(result % 10, 1, "with H1 forces the resumable handler to H1");
}
