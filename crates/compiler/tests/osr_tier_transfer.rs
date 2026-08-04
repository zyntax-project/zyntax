//! Does a tier-1 promotion through the real ladder install OSR entries?
//!
//! `test_osr_helper_emission_for_counted_loop` proves a tier-1
//! `compile_function` *produces* helpers. This goes one step further and
//! drives `TieredBackend`, which is what a running program uses, then asks
//! the bead whether an entry actually landed under the layout's site key —
//! the same question `osr_probe` asks from JIT'd code.

mod common;

use common::counted_loop;
use zyntax_compiler::hir::HirModule;
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// Whether any registered bead holds an OSR entry under `site`.
fn any_bead_has_entry(site: u64) -> bool {
    osr::bead_registry()
        .read()
        .unwrap()
        .values()
        .any(|bead| bead.osr_entry(site).is_some_and(|p| !p.is_null()))
}

/// Promoting to tier 1 through `TieredBackend` should leave an OSR entry
/// the runtime probe can find. If this fails, the install side is the gap;
/// if it passes, the gap is only that tier-0 never emits the probe.
#[test]
fn a_tier1_promotion_installs_an_osr_entry() {
    let (function, header_id) = counted_loop();
    let func_id = function.id;

    let layout =
        osr::osr_layout(&function, header_id).expect("counted loop should have an OSR layout");
    let site = layout.site_key();

    assert!(
        !any_bead_has_entry(site),
        "no OSR entry should exist before promotion"
    );

    let mut module = HirModule::new(InternedString::new_global("osr_test"));
    module.functions.insert(func_id, function);

    let mut config = TieredConfig::default();
    config.verbosity = 2;
    let mut backend = TieredBackend::new(config).expect("tiered backend");
    backend.compile_module(module).expect("tier-0 compile");

    // Back-edges load the helper slot directly, so it must read null while
    // only tier-0 code exists or a loop would branch into nothing.
    let ids: Vec<u64> = osr::bead_registry()
        .read()
        .unwrap()
        .keys()
        .copied()
        .collect();
    assert!(
        ids.iter().all(|id| osr::helper_for(*id, site).is_null()),
        "no helper should be published while only tier-0 code exists"
    );
    backend
        .optimize_function(func_id, OptimizationTier::Standard)
        .expect("force promote to tier 1");

    // Promotion may be queued on a beadie broker thread; poll rather than
    // assuming the compile already ran.
    let mut landed = false;
    for _ in 0..200 {
        if any_bead_has_entry(site) {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        landed,
        "tier-1 promotion should install an OSR entry under site 0x{site:x}"
    );
    assert!(
        ids.iter().any(|id| !osr::helper_for(*id, site).is_null()),
        "installing helpers should publish one into the slot back-edges load"
    );
}

/// The probe's steady-state cost is what decides whether it can stay on.
/// An unarmed back-edge must be a load of the helper slot and a branch —
/// no call into the runtime, because a call that returns into the loop
/// forces caller-saved registers to be treated as clobbered across the
/// whole body.
#[test]
fn an_unarmed_probe_site_costs_a_load_not_a_call() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (function, _header_id) = counted_loop();
    let func_id = function.id;

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_capture_ir(true);
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(0xBEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");

    let (clif, _) = backend.take_captured_ir().expect("captured CLIF");

    assert!(
        clif.contains("load.i64"),
        "the arm check should load the helper slot:\n{clif}"
    );
    assert!(
        !clif.contains("osr_sample_tick"),
        "the per-iteration tick call should be gone:\n{clif}"
    );
    // The only call is the transfer itself, which returns rather than
    // continuing, so `call` and `call_indirect` counts must match.
    let calls = clif.matches("call ").count();
    let indirect = clif.matches("call_indirect").count();
    assert_eq!(
        calls, 0,
        "no direct call should remain in the loop:\n{clif}"
    );
    assert!(
        indirect >= 1,
        "an armed site should still dispatch to the helper:\n{clif}"
    );
}
