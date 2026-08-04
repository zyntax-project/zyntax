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

/// Arguments the last stub transfer observed, plus how many times it ran.
static STUB_ARGS: std::sync::Mutex<Option<[i64; 4]>> = std::sync::Mutex::new(None);

/// Stand-in for a tier-1 helper. Records what the back-edge handed over and
/// returns a sentinel, so a transfer is distinguishable from tier-0 code
/// simply running the loop to completion.
extern "C" fn stub_helper(a: i64, b: i64, c: i64, d: i64) -> i32 {
    *STUB_ARGS.lock().unwrap() = Some([a, b, c, d]);
    999
}

/// A frame running tier-0 code should enter the helper its back-edge points
/// at, hand over the loop's live values, and return the helper's result.
///
/// The helper is published before the call rather than concurrently, so the
/// transfer is deterministic: this is about whether the transfer path is
/// correct, not about promotion timing.
#[test]
fn a_running_frame_transfers_into_the_published_helper() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    const BEAD: u64 = 0xB0A7;
    let (function, header_id) = counted_loop();
    let func_id = function.id;
    let layout =
        osr::osr_layout(&function, header_id).expect("counted loop should have an OSR layout");
    let site = layout.site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize");
    let tier0 = backend.get_function_ptr(func_id).expect("tier-0 pointer");
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };

    // Nothing published: the back-edge loads null and tier-0 runs the loop.
    assert!(osr::helper_for(BEAD, site).is_null());
    assert_eq!(f(10), 45, "tier-0 alone should sum 0..10");
    assert!(
        STUB_ARGS.lock().unwrap().is_none(),
        "no transfer should occur while the slot is null"
    );

    // Published: the first back-edge must hand control to the stub, whose
    // sentinel proves the transfer rather than the loop finishing normally.
    osr::publish_helper(BEAD, site, stub_helper as *mut ());
    assert_eq!(
        f(10),
        999,
        "a published helper should receive the frame and supply the result"
    );

    // The loop is entered with i = 0, sum = 0, and n from the caller, so
    // that is what the header's live-ins should carry across.
    let args = STUB_ARGS.lock().unwrap().expect("stub should have run");
    assert_eq!(
        &args[..3],
        &[0, 0, 10],
        "live-ins should arrive as (i, sum, n); got {args:?}"
    );
}

/// The real tier-1 helper, not a stub, should finish the loop it inherits.
///
/// The stub test above establishes that a published helper does receive the
/// frame, so an answer that matches the scalar result here means the helper
/// resumed from the header and completed correctly rather than the transfer
/// silently not happening.
#[test]
fn the_tier1_helper_completes_the_loop_it_inherits() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    const BEAD: u64 = 0xB0A8;
    let (function, header_id) = counted_loop();
    let func_id = function.id;
    let site = osr::osr_layout(&function, header_id)
        .expect("counted loop should have an OSR layout")
        .site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");

    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize tier 0");
    let tier0 = backend.get_function_ptr(func_id).expect("tier-0 pointer");

    backend.set_compile_tier(1);
    backend
        .compile_function(func_id, &function)
        .expect("tier-1 compile");
    backend.finalize_definitions().expect("finalize tier 1");
    let (helper_site, helper_code) = backend
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(s, _)| *s == site)
        .expect("tier 1 should emit a helper for the loop header");
    assert!(!helper_code.is_null());

    osr::publish_helper(BEAD, helper_site, helper_code);

    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    for (n, expected) in [(10, 45), (100, 4950), (1000, 499_500)] {
        assert_eq!(
            f(n),
            expected,
            "sum 0..{n} through the tier-1 helper should be {expected}"
        );
    }
}

/// Highest loop counter the mid-flight helper has been entered with, or -1
/// if it has not run. It must be the maximum rather than the latest: once a
/// helper is published every subsequent call transfers at i = 0, which would
/// overwrite the mid-flight observation this test exists to make.
static MIDFLIGHT_I: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

/// Stands in for a tier-1 helper: records where the loop had got to, then
/// finishes it the way the original would.
extern "C" fn midflight_helper(i: i64, sum: i64, n: i64, _unused: i64) -> i32 {
    MIDFLIGHT_I.fetch_max(i, std::sync::atomic::Ordering::AcqRel);
    let mut s = sum as i32;
    let mut k = i as i32;
    while k < n as i32 {
        s = s.wrapping_add(k);
        k += 1;
    }
    s
}

/// A helper published while a loop is already running should be picked up
/// on the next back-edge, not only on the next call.
///
/// Every other transfer test publishes before calling, which cannot
/// distinguish "the back-edge observes the slot each iteration" from "the
/// slot is read once on entry". Promotion happens on a broker thread while
/// the frame is live, so this is the case that matters.
#[test]
fn a_helper_published_mid_flight_is_picked_up_by_the_running_loop() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    const BEAD: u64 = 0xD1F5;
    let (function, header_id) = counted_loop();
    let func_id = function.id;
    let site = osr::osr_layout(&function, header_id)
        .expect("counted loop should have an OSR layout")
        .site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize");
    let tier0 = backend.get_function_ptr(func_id).expect("tier-0 pointer") as usize;

    // Long enough that one call spans the publish below, so the transfer
    // lands on a loop already in flight rather than on the next call.
    const N: i32 = 400_000_000;
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    let baseline = f(N);

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let worker_stop = std::sync::Arc::clone(&stop);
    let worker = std::thread::spawn(move || {
        let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
        let mut results = Vec::new();
        while !worker_stop.load(Ordering::Acquire) {
            results.push(f(N));
        }
        results
    });

    // Publish while the worker is mid-loop, then let it run long enough to
    // cross a back-edge and take the new path.
    std::thread::sleep(std::time::Duration::from_millis(50));
    osr::publish_helper(BEAD, site, midflight_helper as *mut ());
    std::thread::sleep(std::time::Duration::from_millis(200));
    stop.store(true, Ordering::Release);
    let results = worker.join().expect("worker should not fault");

    let observed = MIDFLIGHT_I.load(Ordering::Acquire);
    assert!(
        observed >= 0,
        "the running loop should have reached the published helper"
    );
    assert!(
        observed > 0 && observed < N as i64,
        "the transfer should happen mid-loop, not at entry or after the \
         bound; observed i = {observed}"
    );
    assert!(
        results.iter().all(|r| *r == baseline),
        "every run should agree with the un-transferred result {baseline}"
    );
}
