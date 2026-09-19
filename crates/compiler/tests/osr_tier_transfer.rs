#![cfg(feature = "cranelift-backend")]

//! Does a tier-1 promotion through the real ladder install OSR entries?
//!
//! `test_osr_helper_emission_for_counted_loop` proves a tier-1
//! `compile_function` *produces* helpers. This goes one step further and
//! drives `TieredBackend`, which is what a running program uses, then asks
//! the bead whether an entry actually landed under the layout's site key —
//! the same question `osr_probe` asks from JIT'd code.

mod common;

use common::{counted_loop, flagged_counted_loop};
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirPhi, HirTerminator,
    HirType, HirValue, HirValueKind,
};
use zyntax_compiler::osr;
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
use zyntax_typed_ast::InternedString;

/// Direct `call` instructions in a CLIF dump.
///
/// Counted per line, and only where `call` is the opcode. Searching the
/// dump for the text `call` finds it inside `call_indirect` and inside
/// the name of a calling convention, and `windows_fastcall` appears in
/// the signature of every function compiled for Windows.
fn direct_calls(clif: &str) -> usize {
    clif.lines()
        .filter(|line| {
            // Drop the trailing comment first: it can hold ` = `, which
            // would otherwise be read as the result assignment.
            let code = line.split(';').next().unwrap_or_default();
            let opcode = match code.split_once(" = ") {
                Some((_, rhs)) => rhs,
                None => code,
            };
            opcode.trim_start().starts_with("call ")
        })
        .count()
}

/// Whether any registered bead holds an OSR entry under `site`.
fn any_bead_has_entry(site: u64) -> bool {
    osr::bead_registry()
        .read()
        .unwrap()
        .values()
        .any(|bead| bead.osr_entry(site).is_some_and(|p| !p.is_null()))
}

/// The loop exit can also be reached before the loop. A helper entered at
/// the loop header only needs the exit phi's incoming value from that header.
fn loop_with_external_exit_entry() -> (HirFunction, HirId, HirId) {
    let (mut function, header) = counted_loop();
    let entry = function.entry_block;
    let exit = function.blocks[&header].successors[1];
    let sum = function.blocks[&header].phis[1].result;
    let zero = function.blocks[&header].phis[1]
        .incoming
        .iter()
        .find(|(_, pred)| *pred == entry)
        .unwrap()
        .0;
    let one = function
        .values
        .iter()
        .find_map(|(id, value)| {
            matches!(value.kind, HirValueKind::Constant(HirConstant::I32(1))).then_some(*id)
        })
        .unwrap();
    let before_loop = HirId::new();
    let exit_result = HirId::new();
    let condition = HirId::new();
    for (id, ty, kind) in [
        (before_loop, HirType::I32, HirValueKind::Instruction),
        (exit_result, HirType::I32, HirValueKind::Instruction),
        (
            condition,
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(true)),
        ),
    ] {
        function.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: Default::default(),
                span: None,
            },
        );
    }
    let entry_block = function.blocks.get_mut(&entry).unwrap();
    entry_block.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Add,
        result: before_loop,
        ty: HirType::I32,
        left: zero,
        right: one,
    });
    entry_block.terminator = HirTerminator::CondBranch {
        condition,
        true_target: header,
        false_target: exit,
    };
    entry_block.successors.push(exit);
    let exit_block = function.blocks.get_mut(&exit).unwrap();
    exit_block.predecessors.push(entry);
    exit_block.phis.push(HirPhi {
        result: exit_result,
        ty: HirType::I32,
        incoming: vec![(before_loop, entry), (sum, header)],
    });
    exit_block.terminator = HirTerminator::Return {
        values: vec![exit_result],
    };
    (function, header, before_loop)
}

/// A block the loop reaches that is also entered from before the loop
/// resumes with only its in-loop edge: the value the entry edge would
/// have brought is never read by the resumed code, so it is not a
/// live-in.
#[test]
fn a_shared_block_that_reenters_the_loop_resumes_without_the_entry_value() {
    let (mut function, header, before_loop) = loop_with_external_exit_entry();
    let exit = function.blocks[&header].successors[1];
    function.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Branch { target: header };
    let layout = osr::osr_layout(&function, header).expect("the loop resumes at its header");
    assert!(!layout.live_ins.contains(&before_loop));
    assert!(layout.repairs.is_empty());
}

#[test]
fn a_helper_ignores_an_exit_edge_outside_its_reachable_graph() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    const BEAD: u64 = 0xB0AC;
    let (function, header, before_loop) = loop_with_external_exit_entry();
    let id = function.id;
    let layout = osr::osr_layout(&function, header).expect("loop should have an OSR layout");
    assert!(!layout.live_ins.contains(&before_loop));
    let site = layout.site_key();
    let osr_syms = osr::osr_runtime_symbols();
    let mut backend = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    backend.set_compile_tier(0);
    backend.set_compile_bead_id(BEAD);
    backend
        .compile_function(id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize tier 0");
    let tier0 = backend.get_function_ptr(id).expect("tier-0 pointer");
    backend.set_compile_tier(1);
    backend
        .compile_function(id, &function)
        .expect("tier-1 compile");
    backend.finalize_definitions().expect("finalize tier 1");
    let (_, helper) = backend
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(s, _)| *s == site)
        .expect("helper for the loop header");
    osr::publish_helper(BEAD, site, helper);
    let run: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    assert_eq!(run(10), 45);
}

#[cfg(feature = "llvm-backend")]
#[test]
fn an_llvm_helper_ignores_an_exit_edge_outside_its_reachable_graph() {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_backend::LLVMBackend;

    let (function, header, before_loop) = loop_with_external_exit_entry();
    let layout = osr::osr_layout(&function, header).expect("loop should have an OSR layout");
    assert!(!layout.live_ins.contains(&before_loop));
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "osr_external_exit");
    backend
        .compile_osr_helper(&function, &layout)
        .expect("LLVM helper should compile");
    backend
        .module()
        .verify()
        .expect("LLVM helper should verify");
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

/// Counting calls has to survive the Windows calling convention.
///
/// A function compiled for Windows carries `windows_fastcall` in its
/// signature, and the last five characters of that spell `call` with a
/// space after them. Counting the text `call ` reads the signature line
/// as a call site, so a probe that is a bare load looks like it calls
/// into the runtime on Windows and nowhere else.
#[test]
fn counting_calls_ignores_the_calling_convention_and_indirect_calls() {
    let windows = "\
function u0:0(i32) -> i32 windows_fastcall {
    sig1 = (i64) -> i32 windows_fastcall
block0(v2: i32):
    v3 = iconst.i64 0xbead
    call fn0(v3)  ; v3 = 0xbead
block4:
    v16 = call_indirect.i64 sig1, v8(v9)
    return v16
}
";
    assert_eq!(
        direct_calls(windows),
        1,
        "only `call fn0` is a direct call in:\n{windows}"
    );

    let system_v = windows.replace("windows_fastcall", "system_v");
    assert_eq!(
        direct_calls(&system_v),
        direct_calls(windows),
        "the count must not depend on the calling convention"
    );

    assert_eq!(direct_calls("    v1 = call fn0(v0)"), 1);
    assert_eq!(direct_calls("    v1 = call_indirect.i64 sig0, v0(v2)"), 0);
    assert_eq!(direct_calls(""), 0);
}

/// An unarmed back-edge should load the helper slot without calling the
/// runtime, except for the one visit that requests promotion; the entry
/// likewise counts in a memory cell and calls only at its threshold.
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
    // Two gated requests: the entry's, behind its call count, and the
    // loop's, behind its back-edge count.
    let calls_total = direct_calls(&clif);
    assert_eq!(
        calls_total, 2,
        "the only direct calls should be the gated promotion requests:\n{clif}"
    );
    assert!(
        clif.contains("iconst.i64 1024"),
        "missing hot-loop gate:\n{clif}"
    );
    assert!(
        clif.contains("iconst.i64 2048"),
        "missing hot-entry gate:\n{clif}"
    );
    assert!(
        clif.matches("call_indirect").count() >= 1,
        "an armed site should still dispatch to the helper:\n{clif}"
    );
}

static PROMOTION_REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

extern "C" fn count_promotion_request(_bead_id: u64) {
    PROMOTION_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[test]
fn only_a_running_hot_loop_requests_promotion() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    let (function, _) = counted_loop();
    let id = function.id;
    let symbol = [(
        osr::OSR_REQUEST_SYMBOL,
        count_promotion_request as *const u8,
    )];
    let mut backend = CraneliftBackend::with_runtime_symbols(&symbol).expect("backend");
    backend.set_compile_tier(0);
    backend
        .compile_function(id, &function)
        .expect("tier-0 compile");
    backend.finalize_definitions().expect("finalize");
    let entry = backend.get_function_ptr(id).expect("entry");
    let run: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(entry) };

    PROMOTION_REQUESTS.store(0, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(run(100), 4950);
    assert_eq!(
        PROMOTION_REQUESTS.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(run(2000), 1_999_000);
    assert_eq!(
        PROMOTION_REQUESTS.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(run(100), 4950);
    assert_eq!(
        PROMOTION_REQUESTS.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

/// Arguments the last stub transfer observed, plus how many times it ran.
static STUB_ARGS: std::sync::Mutex<Option<[i64; 4]>> = std::sync::Mutex::new(None);

/// Stand-in for a tier-1 helper. Records what the back-edge handed over and
/// returns a sentinel, so a transfer is distinguishable from tier-0 code
/// simply running the loop to completion.
extern "C" fn stub_helper(frame: *mut u8) -> i32 {
    // count_to's live-ins are (i, sum, n), all i32, so they sit at the
    // first three slots of the frame.
    let read =
        |off: usize| unsafe { std::ptr::read_unaligned(frame.add(off) as *const i32) as i64 };
    *STUB_ARGS.lock().unwrap() = Some([read(0), read(4), read(8), 0]);
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

/// A live-in narrower than a word arrives with only its own bytes.
///
/// The flag is a `bool` the frame holds in one byte, ahead of the wider
/// `3 * n`. A helper that read it as a word would take that value's
/// bytes for part of the flag and see it set, adding 1000 a step where
/// the tier-0 code added `3 * n`.
#[test]
fn the_tier1_helper_reads_a_byte_live_in_as_a_byte() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;

    const BEAD: u64 = 0xB0AB;
    let (function, header_id) = flagged_counted_loop();
    let func_id = function.id;
    let layout =
        osr::osr_layout(&function, header_id).expect("flagged loop should have an OSR layout");
    assert!(
        layout
            .live_in_types
            .iter()
            .any(|t| matches!(t, zyntax_compiler::hir::HirType::Bool)),
        "the flag should travel as a bool: {:?}",
        layout.live_in_types
    );
    let site = layout.site_key();

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
    osr::publish_helper(BEAD, helper_site, helper_code);

    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(tier0) };
    for (n, expected) in [(10, 300), (7, 10_000), (100, 3000)] {
        assert_eq!(
            f(n),
            expected,
            "count_flagged({n}) through the tier-1 helper should be {expected}"
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
extern "C" fn midflight_helper(frame: *mut u8) -> i32 {
    let read = |off: usize| unsafe { std::ptr::read_unaligned(frame.add(off) as *const i32) };
    let (i, sum, n) = (read(0), read(4), read(8));
    MIDFLIGHT_I.fetch_max(i as i64, std::sync::atomic::Ordering::AcqRel);
    let mut s = sum;
    let mut k = i;
    while k < n {
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

/// `TieredConfig::enable_osr` should decide whether back-edges carry a
/// probe at all.
///
/// Cold-start serving wants it on — a worker invocation is often one long
/// call that call-count promotion can never reach — while a deployment that
/// warms up first can pay nothing for it.
#[test]
fn the_osr_config_cog_controls_probe_emission() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::tiered_backend::TieredConfig;

    assert!(
        TieredConfig::default().enable_osr,
        "cold start is the default case, so probes default to on"
    );

    let (function, _header) = counted_loop();
    let func_id = function.id;

    for enabled in [true, false] {
        let mut backend = CraneliftBackend::new().expect("backend");
        backend.set_capture_ir(true);
        backend.set_compile_tier(0);
        backend.set_emit_osr_probes(enabled);
        backend
            .compile_function(func_id, &function)
            .expect("tier-0 compile");
        let (clif, _) = backend.take_captured_ir().expect("captured CLIF");

        assert_eq!(
            clif.contains("call_indirect"),
            enabled,
            "enable_osr = {enabled} should decide whether the back-edge can \
             transfer:\n{clif}"
        );
    }
}

/// The frame both backends read and write: scalars at their natural
/// alignment, and an aggregate as the pointer to its storage rather than
/// a copy of it. The storage may be shared (a list header another live-in
/// addresses), so the resumed code has to keep writing where everything
/// else reads.
#[test]
fn the_osr_frame_lays_out_live_ins_at_natural_alignment() {
    use zyntax_compiler::hir::{HirStructType, HirType};
    use zyntax_compiler::osr::OsrFrame;
    use zyntax_typed_ast::InternedString;

    // nbody's Body: seven doubles, held by reference.
    let body = HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Body")),
        fields: vec![HirType::F64; 7],
        packed: false,
    });
    assert!(zyntax_compiler::osr::is_held_by_reference(&body));
    assert_eq!(
        zyntax_compiler::osr::frame_size_of(&body),
        8,
        "an aggregate travels as its pointer"
    );
    assert_eq!(zyntax_compiler::osr::frame_align_of(&body), 8);

    // A mixed frame: each entry starts at its own alignment, and the frame
    // is padded to the widest.
    let frame = OsrFrame::for_types(&[
        HirType::I32,
        HirType::F64,
        body.clone(),
        HirType::I8,
        HirType::I64,
    ]);
    assert_eq!(
        frame.offsets,
        vec![0, 8, 16, 24, 32],
        "i32 at 0, f64 realigned to 8, Body's pointer at 16, i8 after it, i64 realigned"
    );
    assert_eq!(frame.align, 8);
    assert_eq!(frame.size, 40, "padded to the frame's own alignment");

    // A frame of one scalar is just that scalar.
    let single = OsrFrame::for_types(&[HirType::I64]);
    assert_eq!(single.offsets, vec![0]);
    assert_eq!(single.size, 8);

    // No live-ins is a valid, empty frame rather than an error.
    let empty = OsrFrame::for_types(&[]);
    assert_eq!(empty.size, 0);
    assert!(empty.offsets.is_empty());
}

/// A loop that grows a heap header it holds by reference, with the header's
/// length field addressed by a second live-in computed before the loop.
///
/// The helper has to keep writing into the same header the address live-in
/// reads, so the frame must carry the header's pointer rather than a copy
/// of its bytes: with a copy, the resumed loop advances the copy's data
/// pointer while the real header's length keeps growing, which is a list
/// whose length says one thing and whose storage says another.
#[test]
fn a_transfer_keeps_writing_into_the_header_it_was_handed() {
    use indexmap::IndexMap;
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::hir::{
        BinaryOp, HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
        HirParam, HirPhi, HirStructType, HirTerminator, HirType, HirValue, HirValueKind,
    };

    const BEAD: u64 = 0xB0A9;
    let header_ty = HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("List")),
        fields: vec![HirType::I64; 3],
        packed: false,
    });
    let i64_ty = HirType::I64;
    let ptr_i64 = HirType::Ptr(Box::new(HirType::I64));

    let [entry_id, head_id, body_id, exit_id] = [(); 4].map(|_| HirId::new());
    let [
        list,
        n,
        c0,
        c1,
        c8,
        len_addr,
        phi_i,
        cmp,
        data,
        bumped,
        len,
        len1,
        next_i,
    ] = [(); 13].map(|_| HirId::new());

    let mut values: IndexMap<HirId, HirValue> = IndexMap::new();
    let mut value = |id: HirId, ty: HirType, kind: HirValueKind| {
        values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: Default::default(),
                span: None,
            },
        );
    };
    value(list, header_ty.clone(), HirValueKind::Parameter(0));
    value(n, i64_ty.clone(), HirValueKind::Parameter(1));
    for (id, v) in [(c0, 0), (c1, 1), (c8, 8)] {
        value(
            id,
            i64_ty.clone(),
            HirValueKind::Constant(HirConstant::I64(v)),
        );
    }
    for id in [len_addr, data, bumped] {
        value(id, ptr_i64.clone(), HirValueKind::Instruction);
    }
    for id in [phi_i, len, len1, next_i] {
        value(id, i64_ty.clone(), HirValueKind::Instruction);
    }
    value(cmp, HirType::Bool, HirValueKind::Instruction);

    let block = |id, phis, instructions, terminator, predecessors, successors| HirBlock {
        id,
        label: None,
        phis,
        instructions,
        terminator,
        dominance_frontier: Default::default(),
        predecessors,
        successors,
    };
    let mut blocks: IndexMap<HirId, HirBlock> = IndexMap::new();
    blocks.insert(
        entry_id,
        block(
            entry_id,
            vec![],
            vec![HirInstruction::GetElementPtr {
                result: len_addr,
                ty: HirType::U8,
                ptr: list,
                indices: vec![c8],
            }],
            HirTerminator::Branch { target: head_id },
            vec![],
            vec![head_id],
        ),
    );
    blocks.insert(
        head_id,
        block(
            head_id,
            vec![HirPhi {
                result: phi_i,
                ty: i64_ty.clone(),
                incoming: vec![(c0, entry_id), (next_i, body_id)],
            }],
            vec![HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: cmp,
                ty: HirType::Bool,
                left: phi_i,
                right: n,
            }],
            HirTerminator::CondBranch {
                condition: cmp,
                true_target: body_id,
                false_target: exit_id,
            },
            vec![entry_id, body_id],
            vec![body_id, exit_id],
        ),
    );
    blocks.insert(
        body_id,
        block(
            body_id,
            vec![],
            vec![
                // The element goes where the header's data pointer points,
                // then the pointer is bumped and written back through the
                // header, and the length is bumped through its address.
                HirInstruction::Load {
                    result: data,
                    ty: ptr_i64.clone(),
                    ptr: list,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::Store {
                    value: phi_i,
                    ptr: data,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::GetElementPtr {
                    result: bumped,
                    ty: HirType::U8,
                    ptr: data,
                    indices: vec![c8],
                },
                HirInstruction::Store {
                    value: bumped,
                    ptr: list,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::Load {
                    result: len,
                    ty: i64_ty.clone(),
                    ptr: len_addr,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: len1,
                    ty: i64_ty.clone(),
                    left: len,
                    right: c1,
                },
                HirInstruction::Store {
                    value: len1,
                    ptr: len_addr,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: next_i,
                    ty: i64_ty.clone(),
                    left: phi_i,
                    right: c1,
                },
            ],
            HirTerminator::Branch { target: head_id },
            vec![head_id],
            vec![head_id],
        ),
    );
    blocks.insert(
        exit_id,
        block(
            exit_id,
            vec![],
            vec![],
            HirTerminator::Return {
                values: vec![phi_i],
            },
            vec![head_id],
            vec![],
        ),
    );

    let signature = HirFunctionSignature {
        params: vec![
            HirParam {
                id: list,
                name: InternedString::new_global("list"),
                ty: header_ty,
                attributes: Default::default(),
                ownership: Default::default(),
            },
            HirParam {
                id: n,
                name: InternedString::new_global("n"),
                ty: i64_ty.clone(),
                attributes: Default::default(),
                ownership: Default::default(),
            },
        ],
        returns: vec![i64_ty],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut function = HirFunction::new(InternedString::new_global("fill"), signature);
    function.values = values;
    function.blocks = blocks;
    function.entry_block = entry_id;
    function.is_external = false;
    let func_id = function.id;

    let layout = osr::osr_layout(&function, head_id).expect("the loop has a layout");
    let site = layout.site_key();
    assert!(
        layout.live_ins.contains(&list) && layout.live_ins.contains(&len_addr),
        "the header and its length's address both arrive in the frame"
    );

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
    osr::publish_helper(BEAD, helper_site, helper_code);

    // A header over a buffer of eight; the first back edge transfers.
    let count = 8i64;
    let mut buffer = vec![-1i64; count as usize];
    let mut header = [buffer.as_mut_ptr() as i64, 0i64, count];
    let f: extern "C" fn(*mut i64, i64) -> i64 = unsafe { std::mem::transmute(tier0) };
    assert_eq!(f(header.as_mut_ptr(), count), count);
    assert_eq!(header[1], count, "the length grew through its address");
    assert_eq!(
        header[0],
        buffer.as_ptr() as i64 + count * 8,
        "the data pointer grew through the header the helper was handed"
    );
    assert_eq!(buffer, (0..count).collect::<Vec<_>>());
}
