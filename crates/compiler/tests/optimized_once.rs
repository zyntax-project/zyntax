//! A lazy function's HIR goes through the pipeline once, and every tier
//! above the interpreter compiles that body: the baseline, the LLVM tier
//! through the promotion requester, and the LLVM tier through an
//! explicit `optimize_function`.

#![cfg(all(feature = "llvm-backend", feature = "cranelift-backend"))]

mod common;

use std::collections::HashSet;
use std::sync::Arc;

use zyntax_compiler::hir::{HirFunction, HirId, HirModule};
use zyntax_compiler::opt_audit;
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{
    OptimizationTier, Tier2Backend, TieredBackend, TieredConfig,
};
use zyntax_typed_ast::InternedString;

/// `count_to`, left for its first call as the runtime leaves a
/// program's functions.
fn lazy_loop() -> HirFunction {
    let (mut function, _) = common::counted_loop();
    function.attributes.optimized = true;
    function.attributes.deferred = true;
    function
}

fn call(entry: *const u8, n: i32) -> i32 {
    // SAFETY: the entry is `count_to`'s code, `fn(i32) -> i32`.
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(entry) };
    f(n)
}

/// Waits for the LLVM tier to have been handed `id`'s HIR `n` times.
fn llvm_compiles(id: HirId, n: usize) -> Vec<usize> {
    for _ in 0..1000 {
        let bodies = opt_audit::llvm_bodies(id);
        if bodies.len() >= n {
            return bodies;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    opt_audit::llvm_bodies(id)
}

/// One test: the lazy compiler and the requester a backend installs are
/// the process's.
#[test]
fn every_tier_compiles_the_one_optimised_body() {
    opt_audit::enable();
    let (by_requester, by_request) = (lazy_loop(), lazy_loop());
    let (a, b) = (by_requester.id, by_request.id);
    let mut module = HirModule::new(InternedString::new_global("optimized_once"));
    module.functions.insert(a, by_requester);
    module.functions.insert(b, by_request);
    let config = TieredConfig {
        tier2_backend: Tier2Backend::LLVM,
        enable_osr: true,
        ..TieredConfig::default()
    };
    let mut backend = TieredBackend::new(config).expect("tiered backend");
    backend.set_emit_osr_probes(true);
    backend
        .compile_module_lazily(module, None, HashSet::from([a, b]), HashSet::new(), false)
        .expect("module compiles");

    // Each is called through its stub, which optimises the body and
    // compiles it at the baseline; a call too short for its loop to ask
    // for the tier above.
    let (_, entry, bead) = backend.interpreter_bridge();
    for id in [a, b] {
        assert_eq!(call(entry(id).expect("a stub"), 5), 10);
    }
    let (bead_a, bead_b) = (bead(a).expect("a bead"), bead(b).expect("a bead"));
    let stored = |bead| osr::lazy_optimized_body(bead).expect("an optimised body");
    let (body_a, body_b) = (stored(bead_a), stored(bead_b));

    // The requester, as a compiled frame asks.
    assert!(osr::run_promotion(
        bead_a,
        osr::Requester::Compiled { site: osr::NO_SITE }
    ));
    let handed = llvm_compiles(a, 1);
    assert_eq!(
        handed,
        vec![Arc::as_ptr(&body_a) as usize],
        "the requester's LLVM compile took a body other than the optimised one"
    );

    // An explicit promotion.
    backend
        .optimize_function(b, OptimizationTier::Optimized)
        .expect("promotion");
    let handed = llvm_compiles(b, 1);
    assert_eq!(
        handed,
        vec![Arc::as_ptr(&body_b) as usize],
        "optimize_function's LLVM compile took a body other than the optimised one"
    );

    for id in [a, b] {
        assert_eq!(
            opt_audit::pipeline_runs(id),
            1,
            "the pipeline ran more than once over one function"
        );
    }
    // The interpreter runs the same body.
    let mut source = backend.interpreter_body_source();
    let interp = source(a).expect("a body for the interpreter");
    assert!(Arc::ptr_eq(&interp, &body_a));
}
