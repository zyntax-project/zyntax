//! Regional handler dispatch — `with H { }` overrides the static
//! handler for the enclosing effect within its scope.
//!
//! Phase 1 (slices 2-4) of the fiber×effect×async plan: op-table
//! construction + push/pop emission on scope-exit edges +
//! `PerformEffect` dynamic dispatch (`select(lookup_op != 0, dyn,
//! static)` + `call_indirect`). Two handlers implement the same
//! effect; a perform resolves to the static (first-declared) handler
//! outside any `with`, and to the `with`-scoped handler inside.

use zynml::ZynML;

#[test]
fn with_block_overrides_static_handler() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 1 }
        }

        handler H2 for Log {
            def emit(): i64 { return 2 }
        }

        @effect(Log)
        def work(): i64 {
            return emit()
        }

        def main(): i64 {
            let outside = work()      // static first-match -> H1 -> 1
            let mut inside: i64 = 0
            with H2 {
                inside = work()       // with-scoped -> H2 -> 2
            }
            // and back to static after the scope closes
            let after = work()        // H1 again -> 1
            return outside * 100 + inside * 10 + after
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    // outside=1, inside=2, after=1  ->  1*100 + 2*10 + 1 = 121
    assert_eq!(
        result, 121,
        "static H1 outside the scope, H2 inside, H1 again after"
    );
}
