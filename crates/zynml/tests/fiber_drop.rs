//! Phase 2 of the fiber×effect×async plan: fiber drop-site emission.
//! A fiber created (and not moved out) in a function is freed at scope
//! exit via `FiberDrop` -> `krio_fiber_free`. These exercise the paths
//! that would crash on a double-free and confirm correct results.

use zynml::ZynML;

/// Create + fully consume a fiber, then return a non-fiber value. The
/// fiber is dropped at the return; running many of them would crash on
/// a double-free.
#[test]
fn fiber_dropped_at_scope_exit() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        fiber def gen(): i64 {
            yield 1
            yield 2
            yield 3
        }
        def main(): i64 {
            let f = gen()
            let mut sum: i64 = 0
            while let Some(x) = f.next() {
                sum = sum + x
            }
            return sum
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(
        result, 6,
        "1+2+3 = 6; fiber freed at return without corrupting the result"
    );
}

/// A function that RETURNS its fiber must NOT drop it (the caller owns
/// it). If the escape check were wrong this would double-free (free in
/// the producer, then again — or use-after-free — in the consumer).
#[test]
fn returned_fiber_not_double_freed() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        fiber def gen(): i64 {
            yield 10
            yield 20
        }
        def make(): Fiber<i64> {
            let f = gen()
            return f
        }
        def main(): i64 {
            let g = make()
            let mut sum: i64 = 0
            while let Some(x) = g.next() {
                sum = sum + x
            }
            return sum
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(
        result, 30,
        "producer returns the fiber intact; consumer drains it"
    );
}
