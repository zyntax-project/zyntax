//! `@effect(E)` binds to a method, not only to a free function.
//!
//! Effects had only ever been annotated on free functions, and the
//! grammar reflected that: an impl item had no annotation slot at all,
//! so `@effect(Log)` above a method was a parse error. Everything
//! downstream had the same hole — `TypedMethod` carried no annotations,
//! and the impl-block lowering hardcoded an empty effect list on the
//! function it synthesizes per method.
//!
//! These cover the whole path: that the annotation parses, that a
//! perform inside a method body reaches the static handler, that a
//! `with` region still overrides it, and that a method reached through
//! a trait or without a `self` behaves the same.

use zynml::{ZynML, ZynMLConfig, ZynMLRuntimeProfile};

fn run(src: &str) -> i64 {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(src).expect("load");
    rt.call_with_result("main").expect("call")
}

/// The same program under the tiered profile, where the interpreter is
/// the entry tier. It cannot run a perform itself, so the method's
/// function has to be recognised as effectful and routed to the JIT
/// under its mangled name rather than its declared one.
fn run_tiered(src: &str) -> i64 {
    let cfg = ZynMLConfig {
        runtime_profile: ZynMLRuntimeProfile::TieredDevelopment,
        ..ZynMLConfig::default()
    };
    let mut rt = ZynML::with_config(cfg).expect("rt");
    rt.load_source(src).expect("load");
    rt.call_with_result("main").expect("call")
}

/// The base case: a perform in an inherent method resolves to the
/// declared handler, and `self` is still reachable beside it.
#[test]
fn an_inherent_method_performs_its_effect() {
    let out = run(r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 7 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @effect(Log)
            def bump(self): i64 {
                return self.base + emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 10 }
            return c.bump()
        }
    "#);
    assert_eq!(out, 17, "self.base + the handler's 7");
}

/// A `with` region around the call site overrides the static handler
/// for a perform that happens inside a method, the way it already did
/// for one inside a free function.
#[test]
fn a_with_region_overrides_the_handler_inside_a_method() {
    let out = run(r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 1 }
        }

        handler H2 for Log {
            def emit(): i64 { return 2 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @effect(Log)
            def bump(self): i64 {
                return emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 0 }
            let outside = c.bump()
            let mut inside: i64 = 0
            with H2 {
                inside = c.bump()
            }
            let after = c.bump()
            return outside * 100 + inside * 10 + after
        }
    "#);
    assert_eq!(out, 121, "H1 outside the region, H2 inside, H1 again after");
}

/// A method with no `self` is lowered through the same path, so the
/// annotation has to survive there too.
#[test]
fn a_method_without_self_performs_its_effect() {
    let out = run(r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 5 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @effect(Log)
            def standalone(): i64 {
                return emit() * 3
            }
        }

        def main(): i64 {
            return Counter::standalone()
        }
    "#);
    assert_eq!(out, 15);
}

/// Trait impls mangle the method name differently from inherent ones
/// and take a separate branch through the lowering, so they are worth
/// covering on their own.
#[test]
fn a_trait_method_performs_its_effect() {
    let out = run(r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 4 }
        }

        trait Bumpable {
            def bump(self): i64
        }

        struct Counter { base: i64 }

        impl Bumpable for Counter {
            @effect(Log)
            def bump(self): i64 {
                return self.base + emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 100 }
            return c.bump()
        }
    "#);
    assert_eq!(out, 104);
}

/// More than one annotation on a method, and an annotation that is not
/// an effect, both have to pass through without disturbing the effect
/// list.
#[test]
fn other_annotations_sit_beside_the_effect_one() {
    let out = run(r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 9 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @inline
            @effect(Log)
            def bump(self): i64 {
                return emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 0 }
            return c.bump()
        }
    "#);
    assert_eq!(out, 9);
}

/// `@with` never applied handler scoping and is rejected on a free
/// function. Now that a method carries annotations, it has to be
/// rejected there too rather than silently doing nothing.
#[test]
fn a_with_annotation_on_a_method_is_rejected() {
    let mut rt = ZynML::new().expect("rt");
    let err = rt.load_source(
        r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 1 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @with(H1)
            def bump(self): i64 {
                return emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 0 }
            return c.bump()
        }
    "#,
    );
    assert!(
        err.is_err(),
        "a handler-scoping annotation on a method should be rejected, not ignored"
    );
}

/// The tiered profile enters through the interpreter, which routes a
/// perform to the JIT. A method's function is mangled, so this checks
/// the routing recognises it under that name too.
#[test]
fn a_method_performs_its_effect_on_the_tiered_profile() {
    let out = run_tiered(
        r#"
        effect Log {
            def emit(): i64
        }

        handler H1 for Log {
            def emit(): i64 { return 7 }
        }

        struct Counter { base: i64 }

        impl Counter {
            @effect(Log)
            def bump(self): i64 {
                return self.base + emit()
            }
        }

        def main(): i64 {
            let c = Counter { base: 10 }
            return c.bump()
        }
    "#,
    );
    assert_eq!(out, 17);
}

/// A handler carrying state across performs, driven from a method.
/// Stateful handlers have to be entered through a `with` region, so
/// this exercises the region and the method together.
#[test]
fn a_stateful_handler_counts_performs_from_a_method() {
    let out = run(r#"
        effect Ev {
            def next(): i64
        }

        handler Feed for Ev {
            var n: i64 = 0
            def next(): i64 {
                self.n = self.n + 1
                return self.n
            }
        }

        struct Driver { step: i64 }

        impl Driver {
            @effect(Ev)
            def take(self): i64 {
                return next() * self.step
            }
        }

        def main(): i64 {
            let d = Driver { step: 10 }
            let mut total: i64 = 0
            with Feed {
                total = total + d.take()
                total = total + d.take()
                total = total + d.take()
            }
            return total
        }
    "#);
    // 1*10 + 2*10 + 3*10, the handler's counter surviving across the
    // three performs inside the one region.
    assert_eq!(out, 60);
}
