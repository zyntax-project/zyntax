#![cfg(feature = "cranelift-backend")]

use zyntax_compiler::{heap_scalarize, hir::*};
use zyntax_typed_ast::InternedString;

struct Fixture {
    f: HirFunction,
    head: HirId,
    body: HirId,
    exit: HirId,
    current: HirId,
    next: HirId,
    seven: HirId,
}

fn constant(f: &mut HirFunction, n: i64) -> HirId {
    f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(n)))
}
fn instruction(f: &mut HirFunction, ty: HirType) -> HirId {
    f.create_value(ty, HirValueKind::Instruction)
}
fn call(result: Option<HirId>, intrinsic: Intrinsic, args: Vec<HirId>) -> HirInstruction {
    HirInstruction::Call {
        result,
        callee: HirCallable::Intrinsic(intrinsic),
        args,
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    }
}
fn load(result: HirId, ptr: HirId) -> HirInstruction {
    HirInstruction::Load {
        result,
        ptr,
        ty: HirType::I64,
        align: 8,
        volatile: false,
    }
}
fn store(value: HirId, ptr: HirId) -> HirInstruction {
    HirInstruction::Store {
        value,
        ptr,
        align: 8,
        volatile: false,
    }
}
fn binary(op: BinaryOp, result: HirId, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    }
}

fn fixture(n: i64, null: bool) -> Fixture {
    let mut f = HirFunction::new(
        InternedString::new_global("main"),
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        },
    );
    let entry = f.entry_block;
    let head = f.create_block();
    let body = f.create_block();
    let exit = f.create_block();
    let zero = constant(&mut f, 0);
    let one = constant(&mut f, 1);
    let three = constant(&mut f, 3);
    let seven = constant(&mut f, 7);
    let size = constant(&mut f, 8);
    let limit = constant(&mut f, n);
    let ptr = HirType::Ptr(Box::new(HirType::I64));
    let initial = if null {
        f.create_value(
            ptr.clone(),
            HirValueKind::Constant(HirConstant::Null(ptr.clone())),
        )
    } else {
        instruction(&mut f, ptr.clone())
    };
    let current = instruction(&mut f, ptr.clone());
    let next = instruction(&mut f, ptr);
    let counter = instruction(&mut f, HirType::I64);
    let next_counter = instruction(&mut f, HirType::I64);
    let cond = instruction(&mut f, HirType::Bool);
    let old = instruction(&mut f, HirType::I64);
    let new = instruction(&mut f, HirType::I64);
    let result = instruction(&mut f, HirType::I64);
    let b = f.blocks.get_mut(&entry).unwrap();
    if !null {
        b.instructions.extend([
            call(Some(initial), Intrinsic::Malloc, vec![size]),
            store(seven, initial),
        ]);
    }
    b.terminator = HirTerminator::Branch { target: head };
    let b = f.blocks.get_mut(&head).unwrap();
    b.phis = vec![
        HirPhi {
            result: current,
            ty: f.values[&current].ty.clone(),
            incoming: vec![(initial, entry), (next, body)],
        },
        HirPhi {
            result: counter,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (next_counter, body)],
        },
    ];
    b.instructions
        .push(binary(BinaryOp::Lt, cond, counter, limit));
    b.terminator = HirTerminator::CondBranch {
        condition: cond,
        true_target: body,
        false_target: exit,
    };
    let b = f.blocks.get_mut(&body).unwrap();
    if !null {
        b.instructions
            .extend([load(old, current), binary(BinaryOp::Add, new, old, three)]);
    }
    b.instructions.extend([
        call(Some(next), Intrinsic::Malloc, vec![size]),
        store(if null { seven } else { new }, next),
        call(None, Intrinsic::Free, vec![current]),
        binary(BinaryOp::Add, next_counter, counter, one),
    ]);
    b.terminator = HirTerminator::Branch { target: head };
    if null {
        let encoded = instruction(&mut f, HirType::USize);
        let is_null = instruction(&mut f, HirType::Bool);
        f.blocks.get_mut(&exit).unwrap().instructions.extend([
            HirInstruction::Cast {
                result: encoded,
                ty: HirType::USize,
                op: CastOp::PtrToInt,
                operand: current,
            },
            binary(BinaryOp::Eq, is_null, encoded, zero),
            HirInstruction::Cast {
                result,
                ty: HirType::I64,
                op: CastOp::ZExt,
                operand: is_null,
            },
        ]);
    } else {
        f.blocks
            .get_mut(&exit)
            .unwrap()
            .instructions
            .push(load(result, current));
    }
    let b = f.blocks.get_mut(&exit).unwrap();
    b.instructions
        .push(call(None, Intrinsic::Free, vec![current]));
    b.terminator = HirTerminator::Return {
        values: vec![result],
    };
    f.rebuild_cfg_edges();
    Fixture {
        f,
        head,
        body,
        exit,
        current,
        next,
        seven,
    }
}

fn module(f: HirFunction) -> HirModule {
    let mut m = HirModule::new(InternedString::new_global("heap"));
    m.functions.insert(f.id, f);
    m
}
fn execute(m: &HirModule) -> i64 {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let mut backend = CraneliftBackend::new().unwrap();
    backend.compile_module(m).unwrap();
    backend.finalize_definitions().unwrap();
    let ptr = backend
        .get_function_ptr(*m.functions.keys().next().unwrap())
        .unwrap();
    // SAFETY: the fixture has no parameters and returns i64.
    let call: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    call()
}
fn check(mut f: HirFunction, expected: i64, allocations: usize) {
    assert_eq!(execute(&module(f.clone())), expected);
    assert_eq!(heap_scalarize::run_function(&mut f), allocations);
    assert!(
        f.blocks
            .values()
            .all(|b| b.phis.iter().all(|p| !matches!(p.ty, HirType::Ptr(_))))
    );
    assert!(
        f.blocks
            .values()
            .flat_map(|b| &b.instructions)
            .all(|i| !matches!(
                i,
                HirInstruction::Load { .. }
                    | HirInstruction::Store { .. }
                    | HirInstruction::Call { .. }
            ))
    );
    let m = module(f);
    assert_eq!(execute(&m), expected);
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    assert_eq!(
        value_to_i64(&HirInterpreter::new().call(&m, "main", vec![]).unwrap()).unwrap(),
        expected
    );
    #[cfg(feature = "llvm-backend")]
    {
        use inkwell::{OptimizationLevel, context::Context};
        use zyntax_compiler::llvm_backend::LLVMBackend;
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "heap");
        backend.compile_module(&m).unwrap();
        backend.module().verify().unwrap();
        let engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();
        // SAFETY: the fixture's ABI is fn() -> i64.
        let call = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> i64>(&format!(
                    "func_{:?}",
                    m.functions.keys().next().unwrap()
                ))
                .unwrap()
        };
        assert_eq!(
            unsafe { call.call() },
            expected,
            "{}",
            backend.module().print_to_string()
        );
    }
}

#[test]
fn carries_initialized_fields_and_releases_without_allocating() {
    for n in [0, 1, 20] {
        check(fixture(n, false).f, 7 + n * 3, 2);
    }
}
#[test]
fn retains_null_tests_on_zero_and_nonzero_iteration_paths() {
    for n in [0, 1, 20] {
        check(fixture(n, true).f, i64::from(n == 0), 1);
    }
}
fn cast_incoming(a: &mut Fixture, integer_ty: Option<HirType>) {
    let incoming = a.f.blocks[&a.head].phis[0].incoming.clone();
    for (slot, (value, block)) in incoming.into_iter().enumerate() {
        let ptr_ty = a.f.values[&value].ty.clone();
        let middle_ty = integer_ty
            .clone()
            .unwrap_or_else(|| HirType::Ptr(Box::new(HirType::U8)));
        let encoded = instruction(&mut a.f, middle_ty.clone());
        let restored = instruction(&mut a.f, ptr_ty.clone());
        a.f.blocks.get_mut(&block).unwrap().instructions.extend([
            HirInstruction::Cast {
                result: encoded,
                operand: value,
                ty: middle_ty,
                op: if integer_ty.is_some() {
                    CastOp::PtrToInt
                } else {
                    CastOp::Bitcast
                },
            },
            HirInstruction::Cast {
                result: restored,
                operand: encoded,
                ty: ptr_ty,
                op: if integer_ty.is_some() {
                    CastOp::IntToPtr
                } else {
                    CastOp::Bitcast
                },
            },
        ]);
        a.f.blocks.get_mut(&a.head).unwrap().phis[0].incoming[slot].0 = restored;
    }
}

#[test]
fn follows_lossless_pointer_casts_into_phis() {
    for integer_ty in [None, Some(HirType::I64), Some(HirType::USize)] {
        for n in [0, 1, 20] {
            for null in [false, true] {
                let mut a = fixture(n, null);
                cast_incoming(&mut a, integer_ty.clone());
                check(
                    a.f,
                    if null { i64::from(n == 0) } else { 7 + n * 3 },
                    if null { 1 } else { 2 },
                );
            }
        }
    }
}

#[test]
fn rejects_truncated_pointer_encodings_and_escaping_casts() {
    let mut a = fixture(3, false);
    cast_incoming(&mut a, Some(HirType::I16));
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
    let mut a = fixture(3, false);
    cast_incoming(&mut a, Some(HirType::I64));
    let escaped = a.f.blocks[&a.head].phis[0].incoming[1].0;
    a.f.blocks
        .get_mut(&a.body)
        .unwrap()
        .instructions
        .push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("observe".into()),
            args: vec![escaped],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
}

#[test]
fn leaves_escaping_and_mutable_objects_intact() {
    let mut a = fixture(3, false);
    a.f.blocks.get_mut(&a.exit).unwrap().terminator = HirTerminator::Return {
        values: vec![a.current],
    };
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
    let mut a = fixture(3, false);
    a.f.blocks
        .get_mut(&a.body)
        .unwrap()
        .instructions
        .insert(0, store(a.seven, a.current));
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
    let mut a = fixture(3, false);
    a.f.blocks.get_mut(&a.body).unwrap().instructions.insert(
        0,
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("observe".into()),
            args: vec![a.current],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    );
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
}
#[test]
fn leaves_pointer_identity_and_volatile_accesses_intact() {
    let mut a = fixture(3, false);
    let cmp = instruction(&mut a.f, HirType::Bool);
    a.f.blocks
        .get_mut(&a.body)
        .unwrap()
        .instructions
        .push(binary(BinaryOp::Eq, cmp, a.current, a.next));
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
    let mut a = fixture(3, false);
    for i in &mut a.f.blocks.get_mut(&a.body).unwrap().instructions {
        if let HirInstruction::Load { volatile, .. } = i {
            *volatile = true;
        }
    }
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
}
#[test]
fn rejects_uninitialized_and_overlapping_fields() {
    let mut a = fixture(3, false);
    a.f.blocks
        .get_mut(&a.body)
        .unwrap()
        .instructions
        .retain(|i| !matches!(i, HirInstruction::Store { .. }));
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
    let mut a = fixture(3, false);
    let small = instruction(&mut a.f, HirType::I32);
    a.f.blocks.get_mut(&a.exit).unwrap().instructions.insert(
        0,
        HirInstruction::Load {
            result: small,
            ty: HirType::I32,
            ptr: a.current,
            align: 4,
            volatile: false,
        },
    );
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
}
#[test]
fn does_not_snapshot_an_opaque_incoming_pointer() {
    let mut a = fixture(3, false);
    let ptr = a.f.create_value(
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Parameter(0),
    );
    a.f.blocks.get_mut(&a.head).unwrap().phis[0].incoming[0].0 = ptr;
    assert_eq!(heap_scalarize::run_function(&mut a.f), 0);
}

#[test]
fn exposes_a_small_constructor_feeding_a_phi_without_removing_its_effects() {
    let mut ctor = fixture(0, false).f;
    let entry = ctor.entry_block;
    ctor.blocks.retain(|id, _| *id == entry);
    let allocated = ctor.blocks[&entry].instructions[0].result_id().unwrap();
    ctor.signature.returns = vec![ctor.values[&allocated].ty.clone()];
    ctor.blocks
        .get_mut(&entry)
        .unwrap()
        .instructions
        .push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("initialize_external".into()),
            args: vec![allocated],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
    ctor.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
        values: vec![allocated],
    };
    ctor.rebuild_cfg_edges();
    let mut caller = fixture(3, false).f;
    let entry = caller.entry_block;
    let result = caller.blocks[&entry].instructions[0].result_id().unwrap();
    caller.blocks.get_mut(&entry).unwrap().instructions = vec![HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Function(ctor.id),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    }];
    let id = caller.id;
    let mut m = module(ctor);
    m.functions.insert(id, caller);
    assert_eq!(zyntax_compiler::inline::run_module(&mut m).inlined, 1);
    let instructions: Vec<_> = m.functions[&id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .collect();
    assert!(instructions.iter().any(|i| matches!(i, HirInstruction::Call { callee: HirCallable::Symbol(s), .. } if s == "initialize_external")));
    assert!(instructions.iter().any(|i| matches!(
        i,
        HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            ..
        }
    )));
}

#[test]
fn fields_follow_cross_field_recurrences_through_byte_geps() {
    let mut a = fixture(5, false);
    let entry = a.f.entry_block;
    let initial = a.f.blocks[&entry].instructions[0].result_id().unwrap();
    let size = match &a.f.blocks[&entry].instructions[0] {
        HirInstruction::Call { args, .. } => args[0],
        _ => unreachable!(),
    };
    a.f.values.get_mut(&size).unwrap().kind = HirValueKind::Constant(HirConstant::I64(16));
    let offset = constant(&mut a.f, 8);
    let ten = constant(&mut a.f, 10);
    let ty = HirType::Ptr(Box::new(HirType::I64));
    let initial_second = instruction(&mut a.f, ty.clone());
    let current_second = instruction(&mut a.f, ty.clone());
    let next_second = instruction(&mut a.f, ty);
    let previous_second = instruction(&mut a.f, HirType::I64);
    let gep = |result, ptr| HirInstruction::GetElementPtr {
        result,
        ty: HirType::U8,
        ptr,
        indices: vec![offset],
    };
    a.f.blocks
        .get_mut(&entry)
        .unwrap()
        .instructions
        .extend([gep(initial_second, initial), store(ten, initial_second)]);
    let old = a.f.blocks[&a.body].instructions[0].result_id().unwrap();
    let b = a.f.blocks.get_mut(&a.body).unwrap();
    if let HirInstruction::Binary { right, .. } = &mut b.instructions[1] {
        *right = previous_second;
    }
    b.instructions.splice(
        0..0,
        [
            gep(current_second, a.current),
            load(previous_second, current_second),
        ],
    );
    b.instructions
        .extend([gep(next_second, a.next), store(old, next_second)]);
    check(a.f, 106, 2);
}
