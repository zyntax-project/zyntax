//! Counter arithmetic must use consecutive lanes and preserve scalar tails.
#![cfg(feature = "cranelift-backend")]
mod common;

use zyntax_compiler::{auto_vectorize, hir::*, hir_interp::HirInterpreter, value::ZyntaxValue};
use zyntax_typed_ast::InternedString;

// Tiered backends install process-wide lazy-compiler callbacks.
static TIERED_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture(ty: HirType, inclusive: Option<i64>) -> (HirModule, HirId) {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![ty.clone()];
    let mut f = HirFunction::new(InternedString::new_global("counter"), sig);
    for i in 0..2 {
        let id = f.create_value(ty.clone(), HirValueKind::Parameter(i));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global("arg"),
            ty: ty.clone(),
            attributes: Default::default(),
            ownership: Default::default(),
        });
    }
    let constant = |n| {
        if ty == HirType::I32 {
            HirConstant::I32(n as i32)
        } else {
            HirConstant::I64(n)
        }
    };
    let zero = f.create_value(ty.clone(), HirValueKind::Constant(constant(0)));
    let one = f.create_value(ty.clone(), HirValueKind::Constant(constant(1)));
    let n = inclusive
        .map(|n| f.create_value(ty.clone(), HirValueKind::Constant(constant(n))))
        .unwrap_or(f.signature.params[0].id);
    let mask = f.signature.params[1].id;
    let iv = f.create_value(ty.clone(), HirValueKind::Instruction);
    let iv_next = f.create_value(ty.clone(), HirValueKind::Instruction);
    let acc = f.create_value(ty.clone(), HirValueKind::Instruction);
    let acc_next = f.create_value(ty.clone(), HirValueKind::Instruction);
    let term = f.create_value(ty.clone(), HirValueKind::Instruction);
    let result = f.create_value(ty.clone(), HirValueKind::Instruction);
    let condition = f.create_value(HirType::Bool, HirValueKind::Instruction);
    let header = HirId::new();
    let body = HirId::new();
    let exit = HirId::new();
    let entry = f.entry_block;
    f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
    f.blocks.get_mut(&entry).unwrap().successors = vec![header];
    let mut h = HirBlock::new(header);
    h.predecessors = vec![entry, body];
    h.successors = vec![body, exit];
    h.phis = vec![
        HirPhi {
            result: iv,
            ty: ty.clone(),
            incoming: vec![(zero, entry), (iv_next, body)],
        },
        HirPhi {
            result: acc,
            ty: ty.clone(),
            incoming: vec![(zero, entry), (acc_next, body)],
        },
    ];
    h.instructions.push(HirInstruction::Binary {
        result: condition,
        op: if inclusive.is_some() {
            BinaryOp::Le
        } else {
            BinaryOp::Lt
        },
        ty: HirType::Bool,
        left: iv,
        right: n,
    });
    h.terminator = HirTerminator::CondBranch {
        condition,
        true_target: body,
        false_target: exit,
    };
    let mut b = HirBlock::new(body);
    b.predecessors = vec![header];
    b.successors = vec![header];
    b.instructions = vec![
        HirInstruction::Binary {
            result: term,
            op: BinaryOp::Xor,
            ty: ty.clone(),
            left: iv,
            right: mask,
        },
        HirInstruction::Binary {
            result: acc_next,
            op: BinaryOp::Add,
            ty: ty.clone(),
            left: acc,
            right: term,
        },
        HirInstruction::Binary {
            result: iv_next,
            op: BinaryOp::Add,
            ty: ty.clone(),
            left: iv,
            right: one,
        },
    ];
    b.terminator = HirTerminator::Branch { target: header };
    let mut e = HirBlock::new(exit);
    e.predecessors = vec![header];
    // The final counter as well as the reduction must come from the scalar tail.
    e.instructions.push(HirInstruction::Binary {
        result,
        op: BinaryOp::Xor,
        ty,
        left: acc,
        right: iv,
    });
    e.terminator = HirTerminator::Return {
        values: vec![result],
    };
    f.blocks.insert(header, h);
    f.blocks.insert(body, b);
    f.blocks.insert(exit, e);
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("counter"));
    module.functions.insert(id, f);
    (module, id)
}

fn check(tier: &str, mut call: impl FnMut(i64, i64) -> i64, narrow: bool, inclusive: Option<i64>) {
    for n in [-5i64, 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 99] {
        for mask in [
            -1i64,
            7,
            123,
            if narrow { i32::MAX as i64 } else { i64::MAX },
        ] {
            let trips = inclusive.map(|v| v + 1).unwrap_or(n).max(0);
            let mut expected = 0i64;
            for i in 0..trips {
                expected = expected.wrapping_add(i ^ mask);
            }
            expected ^= trips;
            if narrow {
                expected = expected as i32 as i64;
            }
            assert_eq!(
                call(n, mask),
                expected,
                "tier={tier}, n={n}, mask={mask}, narrow={narrow}, inclusive={inclusive:?}"
            );
        }
    }
}

#[test]
fn vector_counters_tails_overflow_and_liveouts_execute_in_every_tier() {
    for narrow in [false, true] {
        for inclusive in [None, Some(-2), Some(0), Some(16), Some(31)] {
            let ty = if narrow { HirType::I32 } else { HirType::I64 };
            let (mut module, id) = fixture(ty, inclusive);
            let pass = auto_vectorize::AutoVectorizePass {
                use_cost_model: false,
                min_trip_count: 0,
                ..Default::default()
            };
            assert_eq!(
                pass.run(module.functions.get_mut(&id).unwrap()).vectorized,
                1
            );
            let mut interp = HirInterpreter::new();
            check(
                "interpreter",
                |n, mask| match interp
                    .call(
                        &module,
                        "counter",
                        vec![ZyntaxValue::Int(n), ZyntaxValue::Int(mask)],
                    )
                    .unwrap()
                {
                    ZyntaxValue::Int(n) => {
                        if narrow {
                            n as i32 as i64
                        } else {
                            n
                        }
                    }
                    other => panic!("{other:?}"),
                },
                narrow,
                inclusive,
            );
            let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
            clif.compile_module(&module).unwrap();
            clif.finalize_definitions().unwrap();
            // SAFETY: both arguments and result have the selected integer width.
            let ptr = clif.get_function_ptr(id).unwrap();
            check(
                "cranelift",
                |n, mask| unsafe {
                    if narrow {
                        let call: unsafe extern "C" fn(i32, i32) -> i32 = std::mem::transmute(ptr);
                        i64::from(call(n as i32, mask as i32))
                    } else {
                        let call: unsafe extern "C" fn(i64, i64) -> i64 = std::mem::transmute(ptr);
                        call(n, mask)
                    }
                },
                narrow,
                inclusive,
            );
            #[cfg(feature = "llvm-backend")]
            {
                let context = inkwell::context::Context::create();
                let mut llvm =
                    zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
                llvm.compile_module(&module).unwrap();
                let ptr = llvm.get_function_pointer(id).unwrap();
                // SAFETY: same fixture signature, with the backend kept alive.
                check(
                    "llvm",
                    |n, mask| unsafe {
                        if narrow {
                            let call: unsafe extern "C" fn(i32, i32) -> i32 =
                                std::mem::transmute(ptr);
                            i64::from(call(n as i32, mask as i32))
                        } else {
                            let call: unsafe extern "C" fn(i64, i64) -> i64 =
                                std::mem::transmute(ptr);
                            call(n, mask)
                        }
                    },
                    narrow,
                    inclusive,
                );
            }
        }
    }
}

#[test]
fn inclusive_bound_overflow_and_loop_local_liveouts_are_rejected() {
    for (ty, max) in [(HirType::I32, i32::MAX as i64), (HirType::I64, i64::MAX)] {
        let (mut module, id) = fixture(ty, Some(max));
        assert_eq!(
            auto_vectorize::run(module.functions.get_mut(&id).unwrap()).vectorized,
            0
        );
    }
    let (mut module, id) = fixture(HirType::I64, None);
    let f = module.functions.get_mut(&id).unwrap();
    let body = f
        .blocks
        .values()
        .find(|b| b.instructions.len() == 3)
        .unwrap();
    let term = match body.instructions[0] {
        HirInstruction::Binary { result, .. } => result,
        _ => unreachable!(),
    };
    let exit = f
        .blocks
        .values_mut()
        .find(|b| matches!(b.terminator, HirTerminator::Return { .. }))
        .unwrap();
    exit.terminator = HirTerminator::Return { values: vec![term] };
    assert_eq!(auto_vectorize::run(f).vectorized, 0);
}

#[test]
fn lazy_counter_vector_width_follows_the_final_native_tier() {
    let _guard = TIERED_TEST.lock().unwrap();
    use std::collections::HashSet;
    use zyntax_compiler::tiered_backend::{Tier2Backend, TieredBackend, TieredConfig};

    for (tier, vectorized) in [
        (Tier2Backend::Cranelift, true),
        #[cfg(feature = "llvm-backend")]
        (Tier2Backend::LLVM, false),
    ] {
        let (mut module, id) = fixture(HirType::I64, Some(99));
        module.functions.get_mut(&id).unwrap().attributes.deferred = true;
        let mut backend = TieredBackend::new(TieredConfig {
            tier2_backend: tier,
            enable_hot_reload: false,
            ..Default::default()
        })
        .unwrap();
        backend
            .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
            .unwrap();
        let (_, _, bead_of) = backend.interpreter_bridge();
        let body = zyntax_compiler::osr::lazy_optimized_body(bead_of(id).unwrap())
            .expect("lazy optimized body");
        assert_eq!(
            body.blocks
                .values()
                .any(|b| b.phis.iter().any(|p| matches!(p.ty, HirType::Vector(..)))),
            vectorized,
            "fixed-width counter vectors belong to the final Cranelift pipeline"
        );
        let mut module = HirModule::new(InternedString::new_global("selected_counter"));
        module.functions.insert(id, (*body).clone());
        let result = HirInterpreter::new()
            .call(
                &module,
                "counter",
                vec![ZyntaxValue::Int(100), ZyntaxValue::Int(7)],
            )
            .unwrap();
        let expected = (0..100).map(|i| i ^ 7).sum::<i64>() ^ 100;
        assert!(matches!(result, ZyntaxValue::Int(value) if value == expected));
    }
}

#[test]
fn interpreted_counter_loops_resume_in_the_final_cranelift_tier() {
    let _guard = TIERED_TEST.lock().unwrap();
    use std::collections::HashSet;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zyntax_compiler::{
        osr,
        tiered_backend::{Tier2Backend, TieredBackend, TieredConfig},
    };

    for vectorized in [false, true] {
        let (mut module, id) = fixture(HirType::I64, None);
        let function = module.functions.get_mut(&id).unwrap();
        if vectorized {
            assert_eq!(auto_vectorize::run(function).vectorized, 1);
        }
        function.attributes.optimized = true;
        function.attributes.deferred = true;
        let mut backend = TieredBackend::new(TieredConfig {
            tier2_backend: Tier2Backend::Cranelift,
            enable_osr: true,
            enable_hot_reload: false,
            baseline_threshold: u32::MAX,
            ..Default::default()
        })
        .unwrap();
        backend
            .compile_module_lazily(module, None, HashSet::from([id]), HashSet::new(), false)
            .unwrap();
        let module = backend.interpreter_module(id).unwrap();
        let (mut thunk, entry, bead_of) = backend.interpreter_bridge();
        let bead = bead_of(id).unwrap();
        assert_eq!(
            osr::published_entry(bead),
            0,
            "entry must start interpreted"
        );
        let transfers = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&transfers);
        let mut interp = HirInterpreter::new();
        interp.register_tick_callback(id, backend.interpreter_tick_callback(id).unwrap());
        interp.set_body_source(backend.interpreter_body_source());
        interp.set_frame_exit_hook(backend.interpreter_frame_exit_hook());
        interp.set_native_bridge(
            Box::new(move |sig| {
                // This fixture has no calls: only a frame transfer needs a thunk.
                observed.fetch_add(1, Ordering::Relaxed);
                thunk(sig)
            }),
            entry,
            bead_of,
        );
        let n = 1_000_003i64;
        let mask = 7i64;
        let expected = (0..n).map(|i| i ^ mask).sum::<i64>() ^ n;
        let result = interp
            .call(
                &module,
                "counter",
                vec![ZyntaxValue::Int(n), ZyntaxValue::Int(mask)],
            )
            .unwrap();
        assert!(
            matches!(result, ZyntaxValue::Int(value) if value == expected),
            "{result:?}"
        );
        assert!(
            transfers.load(Ordering::Relaxed) > 0,
            "loop must leave the interpreter mid-frame"
        );
    }
}
