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

/// `walk(cell: ptr Cell, n: i64) -> i64`, with `Cell { step: i64 }`, and
/// `bump(x: i64) -> i64 { x + 1000 }`:
///
/// ```text
/// entry:  br header
/// header: i = phi [0, entry], [i', latch]
///         sum = phi [0, entry], [sum', latch]
///         cmp = lt i, n; brcond cmp, body, exit
/// body:   step = load i64, cell; s1 = add sum, step
///         rare = eq i, 3; brcond rare, cold, latch
/// cold:   s2 = call bump(s1); br latch
/// latch:  sum' = phi [s1, body], [s2, cold]; i' = add i, 1; br header
/// exit:   return sum
/// ```
///
/// The loop reads through a pointer to a struct it was handed and calls
/// a function on one of its iterations: the shapes a helper has to
/// carry and reach. Returns the module, `walk`'s id, its header and the
/// ids of its live-ins `(i, sum, n, cell)`.
fn pointer_loop_with_a_cold_call() -> (
    zyntax_compiler::hir::HirModule,
    zyntax_compiler::hir::HirId,
    zyntax_compiler::hir::HirId,
    [zyntax_compiler::hir::HirId; 4],
) {
    use indexmap::IndexMap;
    use zyntax_compiler::hir::{
        BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
        HirInstruction, HirModule, HirParam, HirPhi, HirStructType, HirTerminator, HirType,
        HirValue, HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

    let i64_ty = HirType::I64;
    let cell_ty = HirType::Ptr(Box::new(HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Cell")),
        fields: vec![HirType::I64],
        packed: false,
    })));
    let value = |id, ty: &HirType, kind| HirValue {
        id,
        ty: ty.clone(),
        kind,
        uses: Default::default(),
        span: None,
    };
    let block = |id, phis, instructions, terminator| HirBlock {
        id,
        label: None,
        phis,
        instructions,
        terminator,
        dominance_frontier: Default::default(),
        predecessors: vec![],
        successors: vec![],
    };
    let signature = |params: Vec<(HirId, &str, HirType)>| HirFunctionSignature {
        params: params
            .into_iter()
            .map(|(id, name, ty)| HirParam {
                id,
                name: InternedString::new_global(name),
                ty,
                attributes: Default::default(),
                ownership: Default::default(),
            })
            .collect(),
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    // bump(x) = x + 1000
    let x = HirId::new();
    let thousand = HirId::new();
    let bumped = HirId::new();
    let bump_entry = HirId::new();
    let mut bump = HirFunction::new(
        InternedString::new_global("bump"),
        signature(vec![(x, "x", i64_ty.clone())]),
    );
    bump.values = IndexMap::from([
        (x, value(x, &i64_ty, HirValueKind::Parameter(0))),
        (
            thousand,
            value(
                thousand,
                &i64_ty,
                HirValueKind::Constant(HirConstant::I64(1000)),
            ),
        ),
        (bumped, value(bumped, &i64_ty, HirValueKind::Instruction)),
    ]);
    bump.blocks = IndexMap::from([(
        bump_entry,
        block(
            bump_entry,
            vec![],
            vec![HirInstruction::Binary {
                op: BinaryOp::Add,
                result: bumped,
                ty: i64_ty.clone(),
                left: x,
                right: thousand,
            }],
            HirTerminator::Return {
                values: vec![bumped],
            },
        ),
    )]);
    bump.entry_block = bump_entry;
    let bump_id = bump.id;

    // walk(cell, n)
    let [
        cell,
        n,
        zero,
        one,
        three,
        phi_i,
        phi_sum,
        cmp,
        step,
        s1,
        rare,
        s2,
        sum_next,
        i_next,
    ] = std::array::from_fn(|_| HirId::new());
    let [entry, header, body, cold, latch, exit] = std::array::from_fn(|_| HirId::new());
    let mut walk = HirFunction::new(
        InternedString::new_global("walk"),
        signature(vec![
            (cell, "cell", cell_ty.clone()),
            (n, "n", i64_ty.clone()),
        ]),
    );
    walk.values = IndexMap::from([
        (cell, value(cell, &cell_ty, HirValueKind::Parameter(0))),
        (n, value(n, &i64_ty, HirValueKind::Parameter(1))),
        (
            zero,
            value(zero, &i64_ty, HirValueKind::Constant(HirConstant::I64(0))),
        ),
        (
            one,
            value(one, &i64_ty, HirValueKind::Constant(HirConstant::I64(1))),
        ),
        (
            three,
            value(three, &i64_ty, HirValueKind::Constant(HirConstant::I64(3))),
        ),
        (cmp, value(cmp, &HirType::Bool, HirValueKind::Instruction)),
        (rare, value(rare, &HirType::Bool, HirValueKind::Instruction)),
    ]);
    for id in [phi_i, phi_sum, step, s1, s2, sum_next, i_next] {
        walk.values
            .insert(id, value(id, &i64_ty, HirValueKind::Instruction));
    }
    walk.blocks = IndexMap::from([
        (
            entry,
            block(
                entry,
                vec![],
                vec![],
                HirTerminator::Branch { target: header },
            ),
        ),
        (
            header,
            block(
                header,
                vec![
                    HirPhi {
                        result: phi_i,
                        ty: i64_ty.clone(),
                        incoming: vec![(zero, entry), (i_next, latch)],
                    },
                    HirPhi {
                        result: phi_sum,
                        ty: i64_ty.clone(),
                        incoming: vec![(zero, entry), (sum_next, latch)],
                    },
                ],
                vec![HirInstruction::Binary {
                    op: BinaryOp::Lt,
                    result: cmp,
                    ty: HirType::Bool,
                    left: phi_i,
                    right: n,
                }],
                HirTerminator::CondBranch {
                    condition: cmp,
                    true_target: body,
                    false_target: exit,
                },
            ),
        ),
        (
            body,
            block(
                body,
                vec![],
                vec![
                    HirInstruction::Load {
                        result: step,
                        ty: i64_ty.clone(),
                        ptr: cell,
                        align: 8,
                        volatile: false,
                    },
                    HirInstruction::Binary {
                        op: BinaryOp::Add,
                        result: s1,
                        ty: i64_ty.clone(),
                        left: phi_sum,
                        right: step,
                    },
                    HirInstruction::Binary {
                        op: BinaryOp::Eq,
                        result: rare,
                        ty: HirType::Bool,
                        left: phi_i,
                        right: three,
                    },
                ],
                HirTerminator::CondBranch {
                    condition: rare,
                    true_target: cold,
                    false_target: latch,
                },
            ),
        ),
        (
            cold,
            block(
                cold,
                vec![],
                vec![HirInstruction::Call {
                    result: Some(s2),
                    callee: HirCallable::Function(bump_id),
                    args: vec![s1],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                }],
                HirTerminator::Branch { target: latch },
            ),
        ),
        (
            latch,
            block(
                latch,
                vec![HirPhi {
                    result: sum_next,
                    ty: i64_ty.clone(),
                    incoming: vec![(s1, body), (s2, cold)],
                }],
                vec![HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: i_next,
                    ty: i64_ty.clone(),
                    left: phi_i,
                    right: one,
                }],
                HirTerminator::Branch { target: header },
            ),
        ),
        (
            exit,
            block(
                exit,
                vec![],
                vec![],
                HirTerminator::Return {
                    values: vec![phi_sum],
                },
            ),
        ),
    ]);
    walk.entry_block = entry;
    let walk_id = walk.id;

    let mut module = HirModule::new(InternedString::new_global("pointer_loop"));
    module.functions.insert(bump_id, bump);
    module.functions.insert(walk_id, walk);
    (module, walk_id, header, [phi_i, phi_sum, n, cell])
}

/// A loop that calls a function on a cold path and reads through a
/// struct pointer it was handed gets an LLVM resume point, and finishing
/// in it gives the interpreter's answer.
#[test]
fn a_loop_with_a_cold_call_and_a_pointer_live_in_finishes_in_an_llvm_helper() {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    use zyntax_compiler::hir::HirType;
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    use zyntax_compiler::value::ZyntaxValue;

    const BEAD: u64 = 0x9021;
    let (module, walk_id, header, [phi_i, phi_sum, n_id, cell_id]) =
        pointer_loop_with_a_cold_call();
    let walk = &module.functions[&walk_id];
    let layout = osr::osr_layout(walk, header).expect("the loop should have an OSR layout");
    assert!(
        layout
            .live_in_types
            .iter()
            .any(|ty| matches!(ty, HirType::Ptr(inner) if matches!(**inner, HirType::Struct(_)))),
        "fixture should carry a struct pointer live-in: {:?}",
        layout.live_in_types
    );
    let site = layout.site_key();

    let mut cell: [i64; 1] = [7];
    let n: i64 = 100;
    let mut interp = HirInterpreter::new();
    let expected = interp
        .call(
            &module,
            "walk",
            vec![
                ZyntaxValue::Pointer(cell.as_mut_ptr() as *mut u8),
                ZyntaxValue::Int(n),
            ],
        )
        .expect("the interpreter runs walk");
    let expected = value_to_i64(&expected).expect("an integer");
    assert_eq!(expected, n * 7 + 1000);

    let osr_syms = osr::osr_runtime_symbols();
    let mut cranelift = CraneliftBackend::with_runtime_symbols(&osr_syms).expect("backend");
    cranelift.set_compile_tier(0);
    cranelift.set_compile_bead_id(BEAD);
    cranelift.compile_module(&module).expect("tier-0 compile");
    cranelift.finalize_definitions().expect("finalize");
    let tier0 = cranelift.get_function_ptr(walk_id).expect("tier-0 pointer");
    let f: extern "C" fn(*mut i64, i64) -> i64 = unsafe { std::mem::transmute(tier0) };
    assert_eq!(f(cell.as_mut_ptr(), n), expected, "tier 0 alone");

    let context = Context::create();
    let mut llvm = LLVMJitBackend::new(&context).expect("llvm jit backend");
    llvm.set_compile_tier(1);
    llvm.compile_module(&module).expect("llvm compile");
    let (_, helper_site, helper_code) = llvm
        .take_pending_osr_helpers()
        .into_iter()
        .find(|(id, s, _)| *id == walk_id && *s == site)
        .expect("LLVM should make a resume point for a loop with a call and a pointer live-in");
    assert!(!helper_code.is_null());

    // Entered directly from mid-loop state: i = 5 is past the cold
    // iteration, so 95 more steps of 7 land on top of the sum handed in.
    let helper: extern "C" fn(*mut u8) -> i64 = unsafe { std::mem::transmute(helper_code) };
    let mut frame = vec![0u8; layout.frame.size as usize];
    for (slot, id) in layout.live_ins.iter().enumerate() {
        let off = layout.frame.offsets[slot] as usize;
        let word: u64 = if *id == phi_i {
            5
        } else if *id == phi_sum {
            10
        } else if *id == n_id {
            n as u64
        } else if *id == cell_id {
            cell.as_mut_ptr() as u64
        } else {
            panic!("unexpected live-in {id:?}");
        };
        frame[off..off + 8].copy_from_slice(&word.to_ne_bytes());
    }
    assert_eq!(helper(frame.as_mut_ptr()), 10 + 95 * 7);

    // And through the back-edge: the first header visit transfers, and
    // the whole run, cold call included, still gives the interpreter's
    // answer.
    osr::publish_helper(BEAD, helper_site, helper_code);
    assert_eq!(f(cell.as_mut_ptr(), n), expected);
    cell[0] = 3;
    assert_eq!(f(cell.as_mut_ptr(), 50), 50 * 3 + 1000);
}
