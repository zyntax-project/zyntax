//! Phase 4 of the fiber×effect×async plan: the handler stack survives a
//! fiber stack switch. Fibers share the one thread-local handler stack with
//! their caller, so each fiber's frames are lifted into a per-fiber segment
//! on yield/return and re-pushed on the next resume
//! (`__zyntax_effect_fiber_enter`/`_leave`). A `perform` inside a fiber
//! resolves against the handlers active when it was resumed; a fiber's own
//! open handlers neither leak into the caller nor get stranded across yields.

use zynml::ZynML;

/// A `perform` inside a fiber body resolves to the enclosing `with` handler.
#[test]
fn perform_in_fiber_sees_enclosing_handler() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log { def emit(): i64 }
        handler H for Log { def emit(): i64 { return 7 } }

        @effect(Log)
        fiber def gen(): i64 {
            yield emit()
            yield emit()
        }

        def main(): i64 {
            let mut total: i64 = 0
            with H {
                let f = gen()
                while let Some(x) = f.next() {
                    total = total + x
                }
            }
            return total
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(result, 14, "emit() -> 7 on each of two fiber steps = 14");
}

/// A `with` wrapped around the resume site wins: the innermost handler
/// applies to the perform, and the outer handler is back after the scope.
#[test]
fn innermost_with_around_resume_wins() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log { def emit(): i64 }
        handler H1 for Log { def emit(): i64 { return 1 } }
        handler H2 for Log { def emit(): i64 { return 2 } }

        @effect(Log)
        fiber def gen(): i64 {
            yield emit()
        }

        @effect(Log)
        def after(): i64 { return emit() }

        def main(): i64 {
            let mut inside: i64 = 0
            with H1 {
                let f = gen()
                with H2 {
                    while let Some(x) = f.next() {
                        inside = x
                    }
                }
                let a = after()
                return inside * 10 + a
            }
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(result, 21, "perform under H2 -> 2; after() under H1 -> 1");
}

/// A fiber that opens its own handler and is abandoned mid-scope must not
/// leak that handler into the caller: `after()` sees the caller's handler,
/// not the fiber's. Without the per-fiber segment lift this returns 99.
#[test]
fn fiber_handler_does_not_leak_on_partial_drain() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log { def emit(): i64 }
        handler HOut for Log { def emit(): i64 { return 1 } }
        handler HIn for Log { def emit(): i64 { return 9 } }

        @effect(Log)
        fiber def gen(): i64 {
            with HIn {
                yield emit()
                yield emit()
            }
        }

        @effect(Log)
        def after(): i64 { return emit() }

        def main(): i64 {
            with HOut {
                let f = gen()
                let mut first: i64 = 0
                while let Some(v) = f.next() {
                    first = v
                    break
                }
                let a = after()
                return first * 10 + a
            }
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(
        result, 91,
        "fiber's HIn stays with the fiber (first=9); caller's after() sees HOut (1)"
    );
}

/// Two fibers, each with its own handler, resumed interleaved (A, B, A).
/// A's second perform must still resolve to A's handler even though B pushed
/// its own handler in between. This is the case that *requires* per-fiber
/// handler-stack isolation — without it A's second perform sees B's handler
/// and the result is 500 instead of 400.
#[test]
fn interleaved_fibers_keep_their_own_handlers() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Log { def emit(): i64 }
        handler HA for Log { def emit(): i64 { return 100 } }
        handler HB for Log { def emit(): i64 { return 200 } }

        @effect(Log)
        fiber def genA(): i64 {
            with HA {
                yield emit()
                yield emit()
            }
        }

        @effect(Log)
        fiber def genB(): i64 {
            with HB {
                yield emit()
            }
        }

        def main(): i64 {
            let fa = genA()
            let fb = genB()
            let mut r: i64 = 0
            while let Some(v) = fa.next() {
                r = r + v
                break
            }
            while let Some(v) = fb.next() {
                r = r + v
                break
            }
            while let Some(v) = fa.next() {
                r = r + v
                break
            }
            return r
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(
        result, 400,
        "A=100, B=200, A again=100 (A's handler restored past B's) = 400"
    );
}
