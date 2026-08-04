//! Does the LLVM tier emit a usable OSR helper?
//!
//! The Cranelift path can hand a running frame to a helper, but the
//! measured speedup lives in the LLVM tier, which emitted no helper at all.
//! This checks the emitted IR resumes at the loop header rather than at the
//! function's entry.

#![cfg(feature = "llvm-backend")]

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

    // Four i64 slots in, the function's own return type out — the shape a
    // tier-0 back-edge marshals into.
    assert_eq!(
        helper.count_params(),
        osr::OSR_MAX_LIVE_INS as u32,
        "helper should take one slot per live-in cap:\n{ir}"
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
