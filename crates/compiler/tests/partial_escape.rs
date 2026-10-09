#![cfg(feature = "cranelift-backend")]

use zyntax_compiler::{hir::*, partial_escape};
use zyntax_typed_ast::InternedString;

fn function(name: &str, result: HirType) -> HirFunction {
    HirFunction::new(
        InternedString::new_global(name),
        HirFunctionSignature {
            params: vec![],
            returns: vec![result],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        },
    )
}
fn val(f: &mut HirFunction, ty: HirType) -> HirId {
    f.create_value(ty, HirValueKind::Instruction)
}
fn int(f: &mut HirFunction, n: i64) -> HirId {
    f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(n)))
}
fn ptr() -> HirType {
    HirType::Ptr(Box::new(HirType::I64))
}
fn param(f: &mut HirFunction) -> HirId {
    let n = f.signature.params.len();
    let id = f.create_value(ptr(), HirValueKind::Parameter(n as u32));
    f.signature.params.push(HirParam {
        id,
        name: InternedString::new_global("p"),
        ty: ptr(),
        attributes: Default::default(),
        ownership: Default::default(),
    });
    id
}
fn call(r: HirId, callee: HirCallable, args: Vec<HirId>) -> HirInstruction {
    HirInstruction::Call {
        result: Some(r),
        callee,
        args,
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    }
}
fn store(ptr: HirId, value: HirId) -> HirInstruction {
    HirInstruction::Store {
        ptr,
        value,
        align: 8,
        volatile: false,
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
fn binary(result: HirId, op: BinaryOp, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        result,
        op,
        ty: HirType::I64,
        left,
        right,
    }
}

struct Fixture {
    m: HirModule,
    main: HirId,
    head: HirId,
    exit: HirId,
    current: HirId,
}
fn fixture(n: i64, fallback: bool) -> Fixture {
    // The observer writes through one argument and reads through its alias.
    let mut observe = function("observe", HirType::I64);
    let a = param(&mut observe);
    let b = param(&mut observe);
    let one = int(&mut observe, 1);
    let old = val(&mut observe, HirType::I64);
    let changed = val(&mut observe, HirType::I64);
    let result = val(&mut observe, HirType::I64);
    let block = observe.blocks.get_mut(&observe.entry_block).unwrap();
    block.instructions.extend([
        load(old, a),
        binary(changed, BinaryOp::Add, old, one),
        store(a, changed),
        load(result, b),
    ]);
    block.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let mut echo = function("echo", ptr());
    let a = param(&mut echo);
    let five = int(&mut echo, 5);
    let old = val(&mut echo, HirType::I64);
    let changed = val(&mut echo, HirType::I64);
    let block = echo.blocks.get_mut(&echo.entry_block).unwrap();
    block.instructions.extend([
        load(old, a),
        binary(changed, BinaryOp::Add, old, five),
        store(a, changed),
    ]);
    block.terminator = HirTerminator::Return { values: vec![a] };

    let mut f = function("main", HirType::I64);
    let entry = f.entry_block;
    let head = f.create_block();
    let dispatch = f.create_block();
    let hot = f.create_block();
    let cold = f.create_block();
    let latch = f.create_block();
    let exit = f.create_block();
    let initial = val(&mut f, ptr());
    let current = val(&mut f, ptr());
    let fresh = val(&mut f, ptr());
    let escaped = val(&mut f, ptr());
    let next = val(&mut f, ptr());
    let counter = val(&mut f, HirType::I64);
    let increment = val(&mut f, HirType::I64);
    let cond = val(&mut f, HirType::Bool);
    let choice = val(&mut f, HirType::Bool);
    let parity = val(&mut f, HirType::I64);
    let old = val(&mut f, HirType::I64);
    let new = val(&mut f, HirType::I64);
    let result = val(&mut f, HirType::I64);
    let zero = int(&mut f, 0);
    let one = int(&mut f, 1);
    let three = int(&mut f, 3);
    let seven = int(&mut f, 7);
    let eight = int(&mut f, 8);
    let limit = int(&mut f, n);
    let mask = int(&mut f, if fallback { 1 } else { 0 });
    let block = f.blocks.get_mut(&entry).unwrap();
    block.instructions.extend([
        call(
            initial,
            HirCallable::Intrinsic(Intrinsic::Malloc),
            vec![eight],
        ),
        store(initial, seven),
    ]);
    block.terminator = HirTerminator::Branch { target: head };
    let block = f.blocks.get_mut(&head).unwrap();
    block.phis.extend([
        HirPhi {
            result: current,
            ty: ptr(),
            incoming: vec![(initial, entry), (next, latch)],
        },
        HirPhi {
            result: counter,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (increment, latch)],
        },
    ]);
    block
        .instructions
        .push(binary(cond, BinaryOp::Lt, counter, limit));
    block.terminator = HirTerminator::CondBranch {
        condition: cond,
        true_target: dispatch,
        false_target: exit,
    };
    let block = f.blocks.get_mut(&dispatch).unwrap();
    block.instructions.extend([
        binary(parity, BinaryOp::And, counter, mask),
        binary(choice, BinaryOp::Eq, parity, zero),
    ]);
    block.terminator = HirTerminator::CondBranch {
        condition: choice,
        true_target: hot,
        false_target: cold,
    };
    let block = f.blocks.get_mut(&hot).unwrap();
    block.instructions.extend([
        load(old, current),
        binary(new, BinaryOp::Add, old, three),
        call(
            fresh,
            HirCallable::Intrinsic(Intrinsic::Malloc),
            vec![eight],
        ),
        store(fresh, new),
    ]);
    block.terminator = HirTerminator::Branch { target: latch };
    let block = f.blocks.get_mut(&cold).unwrap();
    block
        .instructions
        .push(call(escaped, HirCallable::Function(echo.id), vec![current]));
    block.terminator = HirTerminator::Branch { target: latch };
    let block = f.blocks.get_mut(&latch).unwrap();
    block.phis.push(HirPhi {
        result: next,
        ty: ptr(),
        incoming: vec![(fresh, hot), (escaped, cold)],
    });
    block
        .instructions
        .push(binary(increment, BinaryOp::Add, counter, one));
    block.terminator = HirTerminator::Branch { target: head };
    let block = f.blocks.get_mut(&exit).unwrap();
    block.instructions.push(call(
        result,
        HirCallable::Function(observe.id),
        vec![current, current],
    ));
    block.terminator = HirTerminator::Return {
        values: vec![result],
    };
    f.rebuild_cfg_edges();
    let main = f.id;
    let mut m = HirModule::new(InternedString::new_global("partial_escape"));
    for f in [f, observe, echo] {
        m.add_function(f);
    }
    Fixture {
        m,
        main,
        head,
        exit,
        current,
    }
}

fn execute(a: &Fixture) -> i64 {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let mut backend = CraneliftBackend::new().unwrap();
    backend.compile_module(&a.m).unwrap();
    backend.finalize_definitions().unwrap();
    let ptr = backend.get_function_ptr(a.main).unwrap();
    // SAFETY: the fixture entry has the fn() -> i64 ABI.
    let call: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    call()
}
fn check(mut a: Fixture, expected: i64) {
    assert_eq!(execute(&a), expected);
    assert_eq!(
        partial_escape::run_function(a.m.functions.get_mut(&a.main).unwrap()),
        2
    );
    assert_eq!(execute(&a), expected);
    assert_eq!(
        partial_escape::run_function(a.m.functions.get_mut(&a.main).unwrap()),
        0
    );
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    assert_eq!(
        value_to_i64(&HirInterpreter::new().call(&a.m, "main", vec![]).unwrap()).unwrap(),
        expected
    );
    #[cfg(feature = "llvm-backend")]
    {
        use inkwell::{OptimizationLevel, context::Context};
        use zyntax_compiler::llvm_backend::LLVMBackend;
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "partial_escape");
        backend.compile_module(&a.m).unwrap();
        backend.module().verify().unwrap();
        let engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();
        if let Some(allocate) = backend.module().get_function("zyntax_alloc") {
            engine.add_global_mapping(
                &allocate,
                zyntax_compiler::pool_alloc::zyntax_alloc as *const () as usize,
            );
        }
        // SAFETY: the fixture entry has the fn() -> i64 ABI.
        let call = unsafe {
            engine
                .get_function::<unsafe extern "C" fn() -> i64>(&format!("func_{:?}", a.main))
                .unwrap()
        };
        assert_eq!(unsafe { call.call() }, expected);
    }
}

#[test]
fn scalar_loop_materializes_once_for_duplicate_observer_arguments() {
    for n in [0, 1, 20] {
        check(fixture(n, false), 8 + 3 * n);
    }
}
#[test]
fn opaque_returns_remain_mutable_and_can_reenter_the_scalar_loop() {
    for n in [2, 3, 20] {
        check(fixture(n, true), 8 + 3 * ((n + 1) / 2) + 5 * (n / 2));
    }
}
#[test]
fn keeps_objects_with_live_aliases_after_an_escape() {
    let mut a = fixture(4, true);
    let f = a.m.functions.get_mut(&a.main).unwrap();
    let loaded = val(f, HirType::I64);
    f.blocks
        .get_mut(&a.exit)
        .unwrap()
        .instructions
        .push(load(loaded, a.current));
    f.blocks.get_mut(&a.exit).unwrap().terminator = HirTerminator::Return {
        values: vec![loaded],
    };
    assert_eq!(partial_escape::run_function(f), 0);
    assert_eq!(execute(&a), 24);
}
#[test]
fn keeps_writes_through_a_joined_pointer() {
    let mut a = fixture(4, false);
    let f = a.m.functions.get_mut(&a.main).unwrap();
    let three = int(f, 3);
    f.blocks
        .get_mut(&a.head)
        .unwrap()
        .instructions
        .push(store(a.current, three));
    assert_eq!(partial_escape::run_function(f), 0);
    assert_eq!(execute(&a), 4);
}

#[test]
fn preserves_null_and_pointer_integer_tests_on_zero_trip_paths() {
    for use_null in [false, true] {
        let mut a = two_field_fixture(0, false, ReadBarrier::None);
        let f = a.m.functions.get_mut(&a.main).unwrap();
        let entry = f.entry_block;
        let absent = f.create_block();
        let null_exit = f.create_block();
        let observe = f.create_block();
        let null = f.create_value(ptr(), HirValueKind::Constant(HirConstant::Null(ptr())));
        let condition = f.create_value(
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(use_null)),
        );
        let zero = int(f, 0);
        let answer = int(f, 42);
        let encoded = val(f, HirType::I64);
        let is_null = val(f, HirType::Bool);
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::CondBranch {
            condition,
            true_target: absent,
            false_target: a.head,
        };
        f.blocks.get_mut(&absent).unwrap().terminator = HirTerminator::Branch { target: a.head };
        f.blocks.get_mut(&a.head).unwrap().phis[0]
            .incoming
            .push((null, absent));
        f.blocks.get_mut(&a.head).unwrap().phis[1]
            .incoming
            .push((zero, absent));
        let instructions = std::mem::take(&mut f.blocks.get_mut(&a.exit).unwrap().instructions);
        f.blocks.get_mut(&observe).unwrap().instructions = instructions;
        let term = std::mem::replace(
            &mut f.blocks.get_mut(&a.exit).unwrap().terminator,
            HirTerminator::CondBranch {
                condition: is_null,
                true_target: null_exit,
                false_target: observe,
            },
        );
        f.blocks.get_mut(&observe).unwrap().terminator = term;
        f.blocks.get_mut(&null_exit).unwrap().terminator = HirTerminator::Return {
            values: vec![answer],
        };
        f.blocks.get_mut(&a.exit).unwrap().instructions.extend([
            HirInstruction::Cast {
                result: encoded,
                op: CastOp::PtrToInt,
                operand: a.current,
                ty: HirType::I64,
            },
            binary(is_null, BinaryOp::Eq, encoded, zero),
        ]);
        f.rebuild_cfg_edges();
        check(a, if use_null { 42 } else { 8 });
    }
}

#[test]
fn retains_identity_observations_and_truncated_pointer_encodings() {
    for truncated in [false, true] {
        let mut a = fixture(1, false);
        let f = a.m.functions.get_mut(&a.main).unwrap();
        if truncated {
            let encoded = val(f, HirType::I32);
            f.blocks.get_mut(&a.exit).unwrap().instructions.insert(
                0,
                HirInstruction::Cast {
                    result: encoded,
                    operand: a.current,
                    op: CastOp::PtrToInt,
                    ty: HirType::I32,
                },
            );
        } else {
            let observed = val(f, HirType::Bool);
            f.blocks
                .get_mut(&a.exit)
                .unwrap()
                .instructions
                .insert(0, binary(observed, BinaryOp::Eq, a.current, a.current));
        }
        assert_eq!(partial_escape::run_function(f), 0);
    }
}

#[test]
fn materialization_preserves_pointer_fields_and_their_pointees() {
    let mut a = fixture(6, true);
    let f = a.m.functions.get_mut(&a.main).unwrap();
    let slot = val(f, ptr());
    let thirteen = int(f, 13);
    let count = int(f, 1);
    let bytes = int(f, 16);
    let offset = int(f, 8);
    let mut entry = vec![
        HirInstruction::Alloca {
            result: slot,
            ty: HirType::I64,
            count: Some(count),
            align: 8,
        },
        store(slot, thirteen),
    ];
    entry.extend(std::mem::take(
        &mut f.blocks.get_mut(&f.entry_block).unwrap().instructions,
    ));
    f.blocks.get_mut(&f.entry_block).unwrap().instructions = entry;
    let blocks: Vec<_> = f.blocks.keys().copied().collect();
    for bid in blocks {
        let old = std::mem::take(&mut f.blocks.get_mut(&bid).unwrap().instructions);
        let mut out = Vec::new();
        for mut inst in old {
            if let HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args,
                ..
            } = &mut inst
            {
                args[0] = bytes;
                let root = *r;
                let address = val(f, HirType::Ptr(Box::new(ptr())));
                out.push(inst);
                out.push(HirInstruction::GetElementPtr {
                    result: address,
                    ty: HirType::U8,
                    ptr: root,
                    indices: vec![offset],
                });
                out.push(store(address, slot));
            } else {
                out.push(inst);
            }
        }
        f.blocks.get_mut(&bid).unwrap().instructions = out;
    }
    let observer =
        a.m.functions
            .values_mut()
            .find(|f| f.name.resolve_global().as_deref() == Some("observe"))
            .unwrap();
    let incoming = observer.signature.params[1].id;
    let off = int(observer, 8);
    let address = val(observer, HirType::Ptr(Box::new(ptr())));
    let pointed = val(observer, ptr());
    let loaded = val(observer, HirType::I64);
    let total = val(observer, HirType::I64);
    let old = match &observer.blocks[&observer.entry_block].terminator {
        HirTerminator::Return { values } => values[0],
        _ => unreachable!(),
    };
    let block = observer.blocks.get_mut(&observer.entry_block).unwrap();
    block.instructions.extend([
        HirInstruction::GetElementPtr {
            result: address,
            ty: HirType::U8,
            ptr: incoming,
            indices: vec![off],
        },
        HirInstruction::Load {
            result: pointed,
            ptr: address,
            ty: ptr(),
            align: 8,
            volatile: false,
        },
        load(loaded, pointed),
        binary(total, BinaryOp::Add, old, loaded),
    ]);
    block.terminator = HirTerminator::Return {
        values: vec![total],
    };
    check(a, 45);
}

#[derive(Clone, Copy)]
enum ReadBarrier {
    None,
    Store,
    Call,
    Volatile,
}

fn two_field_fixture(n: i64, fallback: bool, barrier: ReadBarrier) -> Fixture {
    let mut a = fixture(n, fallback);
    let observe =
        a.m.functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some("observe"))
            .unwrap()
            .id;
    let f = a.m.functions.get_mut(&a.main).unwrap();
    let hot =
        f.blocks
            .iter()
            .find_map(|(bid, b)| {
                b.instructions.iter().any(|inst|
        matches!(inst, HirInstruction::Load { ptr, .. } if *ptr == a.current)).then_some(*bid)
            })
            .unwrap();
    let old = f.blocks[&hot].instructions[0].result_id().unwrap();
    let new = f.blocks[&hot].instructions[1].result_id().unwrap();
    let cell = val(f, ptr());
    let two = int(f, 2);
    let seven = int(f, 7);
    let eleven = int(f, 11);
    let one = int(f, 1);
    let offset = int(f, 8);
    let bytes = int(f, 16);
    let before = val(f, HirType::I64);
    let after = val(f, HirType::I64);
    let intermediate = val(f, HirType::I64);
    let address = val(f, ptr());
    let second = val(f, HirType::I64);
    let sum = val(f, HirType::I64);
    let ignored = val(f, HirType::I64);
    let mut entry = vec![
        HirInstruction::Alloca {
            result: cell,
            ty: HirType::I64,
            count: Some(one),
            align: 8,
        },
        store(cell, two),
    ];
    entry.extend(std::mem::take(
        &mut f.blocks.get_mut(&f.entry_block).unwrap().instructions,
    ));
    f.blocks.get_mut(&f.entry_block).unwrap().instructions = entry;
    let blocks: Vec<_> = f.blocks.keys().copied().collect();
    for bid in blocks {
        let mut out = Vec::new();
        for mut inst in std::mem::take(&mut f.blocks.get_mut(&bid).unwrap().instructions) {
            if let HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args,
                ..
            } = &mut inst
            {
                args[0] = bytes;
                let root = *r;
                let field = val(f, ptr());
                out.push(inst);
                out.push(HirInstruction::GetElementPtr {
                    result: field,
                    ptr: root,
                    ty: HirType::U8,
                    indices: vec![offset],
                });
                out.push(store(field, eleven));
            } else {
                out.push(inst);
            }
        }
        f.blocks.get_mut(&bid).unwrap().instructions = out;
    }
    let old_hot = std::mem::take(&mut f.blocks.get_mut(&hot).unwrap().instructions);
    let mut out = vec![
        load(old, a.current),
        load(before, cell),
        binary(intermediate, BinaryOp::Add, old, before),
    ];
    match barrier {
        ReadBarrier::Store => out.push(store(cell, seven)),
        ReadBarrier::Call => out.push(call(
            ignored,
            HirCallable::Function(observe),
            vec![cell, cell],
        )),
        ReadBarrier::None | ReadBarrier::Volatile => {}
    }
    let mut read = load(after, cell);
    if matches!(barrier, ReadBarrier::Volatile) {
        if let HirInstruction::Load { volatile, .. } = &mut read {
            *volatile = true;
        }
    }
    out.extend([
        read,
        HirInstruction::GetElementPtr {
            result: address,
            ptr: a.current,
            ty: HirType::U8,
            indices: vec![offset],
        },
        load(second, address),
        binary(sum, BinaryOp::Add, intermediate, second),
        binary(new, BinaryOp::Add, sum, after),
    ]);
    out.extend(old_hot.into_iter().skip(2));
    f.blocks.get_mut(&hot).unwrap().instructions = out;
    f.rebuild_cfg_edges();
    a
}

#[test]
fn grouped_reads_preserve_dependencies_and_loop_phi_outputs() {
    for fallback in [false, true] {
        let n = 4;
        let a = two_field_fixture(n, fallback, ReadBarrier::None);
        check(a, if fallback { 48 } else { 68 });
        let mut grouped = two_field_fixture(n, fallback, ReadBarrier::None);
        let f = grouped.m.functions.get_mut(&grouped.main).unwrap();
        let blocks = f.blocks.len();
        assert_eq!(partial_escape::run_function(f), 2);
        assert_eq!(f.blocks.len(), blocks + 9);
    }
}

#[test]
fn grouped_reads_stop_at_memory_and_call_effects() {
    for (barrier, expected) in [
        (ReadBarrier::Store, 63),
        (ReadBarrier::Call, 52),
        (ReadBarrier::Volatile, 48),
    ] {
        check(two_field_fixture(4, true, barrier), expected);
    }
}

#[test]
fn grouped_reads_observe_mutated_fields_of_opaque_inputs() {
    let mut a = two_field_fixture(6, true, ReadBarrier::None);
    let echo =
        a.m.functions
            .values_mut()
            .find(|f| f.name.resolve_global().as_deref() == Some("echo"))
            .unwrap();
    let incoming = echo.signature.params[0].id;
    let offset = int(echo, 8);
    let thirteen = int(echo, 13);
    let field = val(echo, ptr());
    echo.blocks
        .get_mut(&echo.entry_block)
        .unwrap()
        .instructions
        .extend([
            HirInstruction::GetElementPtr {
                result: field,
                ptr: incoming,
                ty: HirType::U8,
                indices: vec![offset],
            },
            store(field, thirteen),
        ]);
    check(a, 72);
}
