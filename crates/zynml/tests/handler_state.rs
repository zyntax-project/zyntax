//! Phase 3 of the fiber×effect×async plan: handler state across performs.
//!
//! A `handler H for E { var s: T = init; ... }` gets a synthesized
//! `@reference` state region, allocated fresh at each `with H { }` entry and
//! threaded as an implicit `self` into every op. `self.field` reads/writes
//! persist across performs within the scope; a new scope starts fresh.

use zynml::ZynML;

/// A counter handler whose `n` field survives across three performs in the
/// same `with` scope: 1, 2, 3 → running total 6.
#[test]
fn state_persists_across_performs() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Counter {
            def next(): i64
        }

        handler Seq for Counter {
            var n: i64 = 0
            def next(): i64 {
                self.n = self.n + 1
                return self.n
            }
        }

        @effect(Counter)
        def tick(): i64 {
            return next()
        }

        def main(): i64 {
            let mut total: i64 = 0
            with Seq {
                total = tick()
                total = total + tick()
                total = total + tick()
            }
            return total
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(
        result, 6,
        "n increments 1,2,3 across performs in one scope: 1+2+3 = 6"
    );
}

/// Each `with` scope allocates its own state, so a second scope starts from
/// the field initializer again rather than inheriting the first scope's value.
#[test]
fn state_resets_between_scopes() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        effect Counter {
            def next(): i64
        }

        handler Seq for Counter {
            var n: i64 = 0
            def next(): i64 {
                self.n = self.n + 1
                return self.n
            }
        }

        @effect(Counter)
        def tick(): i64 {
            return next()
        }

        def main(): i64 {
            let mut a: i64 = 0
            with Seq {
                a = tick()
                a = tick()
            }
            let mut b: i64 = 0
            with Seq {
                b = tick()
            }
            return a * 10 + b
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    // first scope: 1 then 2 -> a = 2; second scope resets -> b = 1
    assert_eq!(result, 21, "second scope starts fresh: a=2, b=1 -> 21");
}
