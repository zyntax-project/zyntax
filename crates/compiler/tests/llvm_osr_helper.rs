//! Does the LLVM tier emit a usable OSR helper?
//!
//! The Cranelift path can hand a running frame to a helper, but the
//! measured speedup lives in the LLVM tier, which emitted no helper at all.
//! This checks the emitted IR resumes at the loop header rather than at the
//! function's entry.

#![cfg(all(feature = "llvm-backend", feature = "cranelift-backend"))]

mod common;

use common::counted_loop;
use inkwell::context::Context;
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_compiler::osr;

#[test]
fn the_llvm_tier_emits_a_helper_that_resumes_at_the_header() {
    let (function, header_id) = counted_loop();
    let layout =
        osr::osr_layout(&function, header_id).expect("counted loop should have an OSR layout");

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "osr_test");
    let name = backend
        .compile_osr_helper(&function, &layout)
        .expect("LLVM should emit an OSR helper");

    let ir = backend.module().print_to_string().to_string();
    let helper = backend
        .module()
        .get_function(&name)
        .expect("helper should be in the module");

    // One frame pointer in, the function's own return type out — the shape
    // a tier-0 back-edge hands over.
    assert_eq!(
        helper.count_params(),
        1,
        "helper should take a single frame pointer:\n{ir}"
    );

    // Entry must be the prologue, and it must branch to the header rather
    // than fall into the function's original entry block.
    let entry = helper.get_first_basic_block().expect("helper entry block");
    assert_eq!(
        entry.get_name().to_str().unwrap(),
        "osr_prologue",
        "the prologue should be the entry block:\n{ir}"
    );
    let header_label = format!("bb_{header_id:?}");
    assert!(
        ir.contains(&format!("br label %\"{header_label}\"")),
        "the prologue should jump straight to the loop header:\n{ir}"
    );

    // Each loop-carried phi must take its entry value from the prologue and
    // its other from the back-edge — never from the original preheader,
    // which would restart the loop from its initial values.
    let phis: Vec<&str> = ir
        .lines()
        .map(str::trim)
        .filter(|l| l.contains("= phi "))
        .collect();
    assert_eq!(
        phis.len(),
        layout.phi_count,
        "expected {} loop-carried phis:\n{ir}",
        layout.phi_count
    );
    for phi in &phis {
        assert!(
            phi.contains("%osr_prologue"),
            "phi should take its entry value from the prologue: {phi}"
        );
        assert_eq!(
            phi.matches('[').count(),
            2,
            "phi should have exactly the prologue and back-edge inputs: {phi}"
        );
    }

    // The function's original entry block only led up to the header, so it
    // must not survive into a helper that starts at the header.
    let entry_label = format!("bb_{:?}", function.entry_block);
    assert!(
        !ir.contains(&entry_label),
        "the original entry block should not be in the helper:\n{ir}"
    );

    assert!(
        backend.module().verify().is_ok(),
        "the helper module should verify:\n{ir}"
    );
    if std::env::var_os("DUMP_OSR_IR").is_some() {
        eprintln!("{ir}");
    }
}

/// End to end: a frame running Cranelift tier-0 code should be able to
/// finish inside an LLVM-compiled helper.
///
/// This is the link the gradient depends on — the speedup lives in the LLVM
/// tier, so a resume point that only Cranelift can produce is worth nothing.
#[test]
fn a_cranelift_frame_finishes_inside_an_llvm_helper() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::hir::HirModule;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    use zyntax_typed_ast::InternedString;

    const BEAD: u64 = 0x11FA;
    let (function, header_id) = counted_loop();
    let func_id = function.id;
    let layout =
        osr::osr_layout(&function, header_id).expect("counted loop should have an OSR layout");
    let site = layout.site_key();

    // Tier 0 in Cranelift: the loop, plus a back-edge that loads this
    // site's helper slot.
    let osr_syms = osr::osr_runtime_symbols();
    let mut cranelift = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    cranelift.set_compile_tier(0);
    cranelift.set_compile_bead_id(BEAD);
    cranelift
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    cranelift.finalize_definitions().expect("finalize");
    let tier0 = cranelift.get_function_ptr(func_id).expect("tier-0 pointer");

    // Tier 1 in LLVM: same function, which now also emits the resume point.
    let context = Context::create();
    let mut llvm = LLVMJitBackend::new(&context).expect("llvm jit backend");
    llvm.set_compile_tier(1);
    let mut module = HirModule::new(InternedString::new_global("llvm_osr"));
    module.functions.insert(func_id, function);
    llvm.compile_module(&module).expect("llvm compile");

    let helpers = llvm.take_pending_osr_helpers();
    let (_, helper_site, helper_code) = helpers
        .into_iter()
        .find(|(_, s, _)| *s == site)
        .expect("LLVM should have produced a helper for the loop header");
    assert!(!helper_code.is_null(), "helper should have an address");

    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    assert_eq!(f(10), 45, "tier-0 alone should sum 0..10");

    // Enter the helper directly from mid-loop state. Tier-0 code can only
    // ever start at i = 0, so an answer that accounts for a non-zero
    // starting point could not have come from anywhere else.
    let helper: extern "C" fn(*mut u8) -> i32 = unsafe { std::mem::transmute(helper_code) };
    // Resuming at i = 5 with sum = 10 and n = 100 adds 5..99 to 10.
    let expected_resume: i32 = 10 + (5..100).sum::<i32>();
    let mut frame = vec![0u8; layout.frame.size as usize];
    for (slot, value) in [5i32, 10, 100].iter().enumerate() {
        let off = layout.frame.offsets[slot] as usize;
        frame[off..off + 4].copy_from_slice(&value.to_ne_bytes());
    }
    assert_eq!(
        helper(frame.as_mut_ptr()),
        expected_resume,
        "the helper should continue the loop from the state handed to it"
    );

    // And through the back-edge, the whole loop still produces the same
    // answers as running it entirely in tier-0 code.
    osr::publish_helper(BEAD, helper_site, helper_code);
    for (n, expected) in [(10, 45), (100, 4950), (1000, 499_500)] {
        assert_eq!(
            f(n),
            expected,
            "sum 0..{n} finished in the LLVM helper should be {expected}"
        );
    }
}

/// Highest counter the running loop was at when it entered LLVM code.
static LLVM_MIDFLIGHT_I: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

/// Address of the LLVM-compiled helper the shim forwards to.
static LLVM_HELPER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Published in the helper's place so the transfer is observable. Records
/// where the loop had reached, then hands the frame to the real LLVM
/// helper — which therefore runs on the worker thread, not the one that
/// compiled it.
extern "C" fn recording_shim(frame: *mut u8) -> i32 {
    // The loop counter is the first live-in, so it sits at offset 0.
    let i = unsafe { std::ptr::read_unaligned(frame as *const i32) };
    LLVM_MIDFLIGHT_I.fetch_max(i as i64, std::sync::atomic::Ordering::AcqRel);
    let addr = LLVM_HELPER.load(std::sync::atomic::Ordering::Acquire);
    let helper: extern "C" fn(*mut u8) -> i32 = unsafe { std::mem::transmute(addr) };
    helper(frame)
}

/// A real LLVM compile landing while a loop is in flight, with the code
/// executed from a different thread than compiled it.
///
/// The mid-flight Cranelift test uses a Rust stub as the helper, so it
/// proves the back-edge re-reads its slot but not that MCJIT code is
/// callable from another thread — the concern the object-file backend was
/// originally built around. Promotion runs on a broker thread, so that is
/// the shape production would hit.
#[test]
fn an_llvm_compile_lands_on_a_loop_running_in_another_thread() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::hir::HirModule;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    use zyntax_typed_ast::InternedString;

    const BEAD: u64 = 0x50AC;
    let (function, header_id) = counted_loop();
    let func_id = function.id;
    let site = osr::osr_layout(&function, header_id)
        .expect("counted loop should have an OSR layout")
        .site_key();

    let osr_syms = osr::osr_runtime_symbols();
    let mut cranelift = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    cranelift.set_compile_tier(0);
    cranelift.set_compile_bead_id(BEAD);
    cranelift
        .compile_function(func_id, &function)
        .expect("tier-0 compile");
    cranelift.finalize_definitions().expect("finalize");
    let tier0 = cranelift.get_function_ptr(func_id).expect("tier-0 pointer") as usize;

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

    // Compile and publish on this thread while the worker runs on another,
    // so the code MCJIT emits here is entered there.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let context = Context::create();
    let mut llvm = LLVMJitBackend::new(&context).expect("llvm jit backend");
    llvm.set_compile_tier(1);
    let mut module = HirModule::new(InternedString::new_global("llvm_osr_soak"));
    module.functions.insert(func_id, function);
    llvm.compile_module(&module).expect("llvm compile");
    let (_, helper_site, helper_code) = llvm
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(_, s, _)| *s == site)
        .expect("LLVM should have produced a helper");
    assert_eq!(helper_site, site);
    LLVM_HELPER.store(helper_code as usize, Ordering::Release);
    osr::publish_helper(BEAD, helper_site, recording_shim as *mut ());

    std::thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, Ordering::Release);
    let results = worker.join().expect("worker should not fault in LLVM code");

    let observed = LLVM_MIDFLIGHT_I.load(Ordering::Acquire);
    assert!(
        observed > 0 && observed < N as i64,
        "the loop should have entered LLVM code mid-flight; observed i = {observed}"
    );
    assert!(
        results.iter().all(|r| *r == baseline),
        "every run, including those finished in LLVM code, should agree with \
         the un-transferred result {baseline}"
    );
}

/// An aggregate is where the backends stop agreeing: one holds it as a
/// pointer, the other as a value, and a header phi can want either — the
/// declared type alone does not say which. Seeding one from the frame with
/// the wrong shape produces a helper LLVM rejects, so this pins the emitted
/// IR to whatever the phi it feeds actually asks for.
#[test]
fn a_helper_seeds_a_loop_carried_aggregate_with_the_shape_its_phi_wants() {
    let (function, header_id) = common::aggregate_carrying_loop();
    let layout = osr::osr_layout(&function, header_id)
        .expect("an aggregate-carrying loop should still have an OSR layout");
    assert!(
        layout
            .live_in_types
            .iter()
            .any(zyntax_compiler::osr::is_held_by_reference),
        "fixture should carry a live-in the backends hold by reference"
    );

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "osr_aggregate_test");
    // `compile_osr_helper` runs the verifier and reports a shape mismatch
    // as an error rather than emitting IR that cannot be installed.
    let name = backend
        .compile_osr_helper(&function, &layout)
        .expect("LLVM should emit a helper for an aggregate-carrying loop");

    let helper = backend
        .module()
        .get_function(&name)
        .expect("helper should be in the module");
    assert!(
        helper.verify(false),
        "helper must verify:\n{}",
        backend.module().print_to_string().to_string()
    );
}
