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
    let site = osr::osr_layout(&function, header_id)
        .expect("counted loop should have an OSR layout")
        .site_key();

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
    llvm.set_compile_bead_id(BEAD);
    let mut module = HirModule::new(InternedString::new_global("llvm_osr"));
    module.functions.insert(func_id, function);
    llvm.compile_module(&module).expect("llvm compile");

    let helpers = llvm.take_pending_osr_helpers();
    let (helper_site, helper_code) = helpers
        .into_iter()
        .find(|(s, _)| *s == site)
        .expect("LLVM should have produced a helper for the loop header");
    assert!(!helper_code.is_null(), "helper should have an address");

    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    assert_eq!(f(10), 45, "tier-0 alone should sum 0..10");

    // Enter the helper directly from mid-loop state. Tier-0 code can only
    // ever start at i = 0, so an answer that accounts for a non-zero
    // starting point could not have come from anywhere else.
    let helper: extern "C" fn(i64, i64, i64, i64) -> i32 =
        unsafe { std::mem::transmute(helper_code) };
    // Resuming at i = 5 with sum = 10 and n = 100 adds 5..99 to 10.
    let expected_resume: i32 = 10 + (5..100).sum::<i32>();
    assert_eq!(
        helper(5, 10, 100, 0),
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
