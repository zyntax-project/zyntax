#![cfg(all(feature = "llvm-backend", feature = "cranelift-backend"))]

mod common;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use zyntax_compiler::hir::HirModule;
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{
    OptimizationTier, Tier2Backend, TieredBackend, TieredConfig,
};
use zyntax_typed_ast::InternedString;

#[test]
fn a_saved_stub_uses_the_promoted_entry() {
    let mut module = HirModule::new(InternedString::new_global("promoted_stub"));
    let mut ids = Vec::new();
    for name in [
        "explicit_promotion",
        "requested_promotion",
        "loop_promotion",
    ] {
        let (mut function, header) = common::counted_loop();
        function.name = InternedString::new_global(name);
        function.attributes.optimized = true;
        function.attributes.deferred = true;
        let site = osr::osr_layout(&function, header).unwrap().site_key();
        ids.push((function.id, site));
        module.functions.insert(function.id, function);
    }
    let mut backend = TieredBackend::new(TieredConfig {
        tier2_backend: Tier2Backend::LLVM,
        enable_osr: true,
        ..TieredConfig::default()
    })
    .expect("tiered backend");
    backend
        .compile_module_lazily(
            module,
            None,
            ids.iter().map(|(id, _)| *id).collect::<HashSet<_>>(),
            ids.iter().map(|(id, _)| *id).collect::<HashSet<_>>(),
            false,
        )
        .expect("lazy module");

    let (_, _, bead_of) = backend.interpreter_bridge();
    for (index, (id, site)) in ids.into_iter().enumerate() {
        let bead = bead_of(id).expect("function bead");
        let stub = backend.get_function_pointer(id).expect("uncompiled stub");
        // SAFETY: the saved entry has the fixture's fn(i32) -> i32 ABI.
        let call: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(stub) };
        assert_eq!(call(5), 10);
        let baseline = osr::published_entry(bead);
        assert_ne!(baseline, 0);
        if index == 0 {
            backend
                .optimize_function(id, OptimizationTier::Optimized)
                .expect("queue LLVM promotion");
        } else if index == 2 {
            osr::osr_request_promotion(bead, site);
        } else {
            assert!(osr::run_promotion(
                bead,
                osr::Requester::Compiled { site: osr::NO_SITE }
            ));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let promoted = loop {
            if let Some(entry) = backend.promoted_function_pointer(id)
                && entry as usize != baseline
            {
                break entry as usize;
            }
            assert!(Instant::now() < deadline, "LLVM promotion did not install");
            std::thread::sleep(Duration::from_millis(5));
        };

        if index == 2 {
            let helper = osr::helper_for(bead, site);
            assert!(!helper.is_null());
            assert!(osr::is_llvm_helper(helper as usize));
        }
        // SAFETY: this stable counter is written by baseline code on this thread.
        let counter = unsafe { &*(osr::entry_counter_addr(bead) as *const AtomicU64) };
        let before = counter.load(Ordering::Relaxed);
        for _ in 0..5 {
            assert_eq!(call(5), 10);
        }
        assert_eq!(
            counter.load(Ordering::Relaxed),
            before,
            "the saved stub still entered baseline code after LLVM promotion"
        );
        assert_eq!(osr::published_entry(bead), promoted);

        // Beads beyond the atomic table reach the runtime's fallback.
        osr::set_published_entry(bead, 0);
        assert_eq!(osr::try_lazy_compile(bead) as usize, promoted);
        assert_eq!(call(5), 10);
        assert_eq!(counter.load(Ordering::Relaxed), before);
        osr::set_published_entry(bead, promoted);
    }
}
