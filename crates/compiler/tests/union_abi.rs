//! Tagged values keep their bytes across copies, returns and native tier changes.

use zyntax_compiler::hir::*;
use zyntax_typed_ast::InternedString;

fn union(tag: HirType, payload: HirType, c_union: bool) -> HirType {
    HirType::Union(Box::new(HirUnionType {
        name: None,
        discriminant_type: Box::new(tag),
        variants: vec![
            HirUnionVariant {
                name: InternedString::new_global("None"),
                ty: HirType::Void,
                discriminant: 7,
            },
            HirUnionVariant {
                name: InternedString::new_global("Value"),
                ty: payload,
                discriminant: 9,
            },
        ],
        is_c_union: c_union,
    }))
}
fn function(name: &str, params: Vec<HirType>, returns: Vec<HirType>) -> HirFunction {
    HirFunction::new(
        InternedString::new_global(name),
        HirFunctionSignature {
            params: params
                .into_iter()
                .map(|ty| HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global("p"),
                    ty,
                    attributes: Default::default(),
                    ownership: Default::default(),
                })
                .collect(),
            returns,
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
fn value(f: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    f.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            span: None,
        },
    );
    id
}
fn constant(f: &mut HirFunction, n: f64) -> HirId {
    value(f, HirType::F64, HirValueKind::Constant(HirConstant::F64(n)))
}
fn emit(f: &mut HirFunction, instruction: HirInstruction) {
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .push(instruction);
}
fn ret(f: &mut HirFunction, v: HirId) {
    f.blocks.get_mut(&f.entry_block).unwrap().terminator =
        HirTerminator::Return { values: vec![v] };
}
fn call(f: &mut HirFunction, id: HirId, args: Vec<HirId>, ty: HirType) -> HirId {
    let result = value(f, ty, HirValueKind::Instruction);
    emit(
        f,
        HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(id),
            args,
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    );
    result
}
fn payload(f: &mut HirFunction, u: HirId) -> HirId {
    let result = value(f, HirType::F64, HirValueKind::Instruction);
    emit(
        f,
        HirInstruction::ExtractUnionValue {
            result,
            ty: HirType::F64,
            union_val: u,
            variant_index: 1,
        },
    );
    result
}
fn binary(f: &mut HirFunction, op: BinaryOp, a: HirId, b: HirId) -> HirId {
    let result = value(f, HirType::F64, HirValueKind::Instruction);
    emit(
        f,
        HirInstruction::Binary {
            op,
            result,
            ty: HirType::F64,
            left: a,
            right: b,
        },
    );
    result
}

fn module(tag: HirType, c_union: bool) -> (HirModule, HirId, HirId, HirId) {
    let ty = union(tag.clone(), HirType::F64, c_union);
    let mut make = function("make", vec![HirType::F64], vec![ty.clone()]);
    let p = value(&mut make, HirType::F64, HirValueKind::Parameter(0));
    let result = value(&mut make, ty.clone(), HirValueKind::Instruction);
    emit(
        &mut make,
        HirInstruction::CreateUnion {
            result,
            union_ty: ty.clone(),
            variant_index: 1,
            value: p,
        },
    );
    ret(&mut make, result);
    let mut echo = function("echo", vec![ty.clone()], vec![ty.clone()]);
    let p = value(&mut echo, ty.clone(), HirValueKind::Parameter(0));
    ret(&mut echo, p);
    let mut main = function("score", vec![], vec![HirType::F64]);
    let a = constant(&mut main, 1.5);
    let b = constant(&mut main, 2.5);
    let p = call(&mut main, make.id, vec![a], ty.clone());
    let q = call(&mut main, make.id, vec![b], ty.clone());
    let e = call(&mut main, echo.id, vec![p], ty.clone());
    let ptr = value(
        &mut main,
        HirType::Ptr(std::sync::Arc::new(ty.clone())),
        HirValueKind::Instruction,
    );
    emit(
        &mut main,
        HirInstruction::Alloca {
            result: ptr,
            ty: ty.clone(),
            count: None,
            align: 8,
        },
    );
    emit(
        &mut main,
        HirInstruction::Store {
            value: p,
            ptr,
            align: 8,
            volatile: false,
        },
    );
    let saved = value(&mut main, ty.clone(), HirValueKind::Instruction);
    emit(
        &mut main,
        HirInstruction::Load {
            result: saved,
            ty: ty.clone(),
            ptr,
            align: 8,
            volatile: false,
        },
    );
    emit(
        &mut main,
        HirInstruction::Store {
            value: q,
            ptr,
            align: 8,
            volatile: false,
        },
    );
    let p = payload(&mut main, p);
    let q = payload(&mut main, q);
    let e = payload(&mut main, e);
    let saved_payload = payload(&mut main, saved);
    let k = constant(&mut main, 1000.0);
    let p = binary(&mut main, BinaryOp::Mul, p, k);
    let k = constant(&mut main, 100.0);
    let q = binary(&mut main, BinaryOp::Mul, q, k);
    let k = constant(&mut main, 10.0);
    let e = binary(&mut main, BinaryOp::Mul, e, k);
    let sum = binary(&mut main, BinaryOp::Add, p, q);
    let sum = binary(&mut main, BinaryOp::Add, sum, e);
    let mut sum = binary(&mut main, BinaryOp::Add, sum, saved_payload);
    if !c_union {
        let disc = value(&mut main, tag, HirValueKind::Instruction);
        emit(
            &mut main,
            HirInstruction::GetUnionDiscriminant {
                result: disc,
                union_val: saved,
            },
        );
        let tag = value(&mut main, HirType::F64, HirValueKind::Instruction);
        emit(
            &mut main,
            HirInstruction::Cast {
                op: CastOp::UiToFp,
                result: tag,
                ty: HirType::F64,
                operand: disc,
            },
        );
        sum = binary(&mut main, BinaryOp::Add, sum, tag);
    }
    ret(&mut main, sum);
    let ids = (main.id, make.id, echo.id);
    let mut module = HirModule::new(InternedString::new_global("union_abi"));
    for f in [main, make, echo] {
        module.functions.insert(f.id, f);
    }
    (module, ids.0, ids.1, ids.2)
}

fn check_all(module: &HirModule, id: HirId, expected: f64) {
    let name = module.functions[&id].name.resolve_global().unwrap();
    let mut interp = zyntax_compiler::hir_interp::HirInterpreter::new();
    assert_eq!(
        interp.call(module, &name, vec![]).unwrap(),
        zyntax_compiler::value::ZyntaxValue::Float(expected)
    );
    #[cfg(feature = "cranelift-backend")]
    {
        let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
        clif.compile_module(module).unwrap();
        clif.finalize_definitions().unwrap();
        for function in module.functions.values().filter(|f| !f.is_external) {
            assert!(
                clif.get_function_ptr(function.id).is_some(),
                "Cranelift did not compile {:?}",
                function.name.resolve_global()
            );
        }
        let run: unsafe extern "C" fn() -> f64 =
            unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
        assert_eq!(unsafe { run() }, expected, "Cranelift {name}");
    }
    #[cfg(feature = "llvm-backend")]
    {
        let context = inkwell::context::Context::create();
        let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
        llvm.set_use_mcjit(true);
        llvm.compile_module(module).unwrap();
        let run: unsafe extern "C" fn() -> f64 =
            unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
        assert_eq!(unsafe { run() }, expected, "LLVM {name}");
    }
}

#[test]
fn module_calls_preserve_union_values() {
    let (module, main, ..) = module(HirType::U8, false);
    check_all(&module, main, 1775.5);
}

#[test]
fn empty_variants_do_not_read_a_payload() {
    let ty = union(HirType::U8, HirType::F64, false);
    let mut make = function("none", vec![], vec![ty.clone()]);
    let dummy = value(&mut make, HirType::Void, HirValueKind::Undef);
    let result = value(&mut make, ty.clone(), HirValueKind::Instruction);
    emit(
        &mut make,
        HirInstruction::CreateUnion {
            result,
            union_ty: ty.clone(),
            variant_index: 0,
            value: dummy,
        },
    );
    ret(&mut make, result);
    let mut main = function("noneScore", vec![], vec![HirType::F64]);
    let u = call(&mut main, make.id, vec![], ty);
    let tag = value(&mut main, HirType::U8, HirValueKind::Instruction);
    emit(
        &mut main,
        HirInstruction::GetUnionDiscriminant {
            result: tag,
            union_val: u,
        },
    );
    let result = value(&mut main, HirType::F64, HirValueKind::Instruction);
    emit(
        &mut main,
        HirInstruction::Cast {
            op: CastOp::UiToFp,
            result,
            ty: HirType::F64,
            operand: tag,
        },
    );
    ret(&mut main, result);
    let id = main.id;
    let mut module = HirModule::new(InternedString::new_global("empty_unions"));
    for f in [main, make] {
        module.functions.insert(f.id, f);
    }
    check_all(&module, id, 7.0);
}

#[test]
fn aggregate_payloads_fill_their_storage() {
    for width in [3, 17] {
        let record = HirType::Struct(HirStructType {
            name: None,
            fields: vec![HirType::F64; width as usize].into(),
            packed: false,
        });
        let ty = union(HirType::U8, record.clone(), false);
        let mut make = function("wide", vec![HirType::F64], vec![ty.clone()]);
        let n = value(&mut make, HirType::F64, HirValueKind::Parameter(0));
        let mut aggregate = value(&mut make, record.clone(), HirValueKind::Undef);
        for index in 0..width {
            let k = constant(&mut make, index as f64);
            let field = binary(&mut make, BinaryOp::Add, n, k);
            let result = value(&mut make, record.clone(), HirValueKind::Instruction);
            emit(
                &mut make,
                HirInstruction::InsertValue {
                    result,
                    ty: record.clone(),
                    aggregate,
                    value: field,
                    indices: vec![index],
                },
            );
            aggregate = result;
        }
        let result = value(&mut make, ty.clone(), HirValueKind::Instruction);
        emit(
            &mut make,
            HirInstruction::CreateUnion {
                result,
                union_ty: ty.clone(),
                variant_index: 1,
                value: aggregate,
            },
        );
        ret(&mut make, result);
        let mut main = function("wideScore", vec![], vec![HirType::F64]);
        let a = constant(&mut main, 3.0);
        let b = constant(&mut main, 5.0);
        let a = call(&mut main, make.id, vec![a], ty.clone());
        let b = call(&mut main, make.id, vec![b], ty);
        let mut fields = Vec::new();
        for u in [a, b] {
            let aggregate = value(&mut main, record.clone(), HirValueKind::Instruction);
            emit(
                &mut main,
                HirInstruction::ExtractUnionValue {
                    result: aggregate,
                    ty: record.clone(),
                    union_val: u,
                    variant_index: 1,
                },
            );
            let result = value(&mut main, HirType::F64, HirValueKind::Instruction);
            emit(
                &mut main,
                HirInstruction::ExtractValue {
                    result,
                    ty: HirType::F64,
                    aggregate,
                    indices: vec![width - 1],
                },
            );
            fields.push(result);
        }
        let hundred = constant(&mut main, 100.0);
        let a = binary(&mut main, BinaryOp::Mul, fields[0], hundred);
        let result = binary(&mut main, BinaryOp::Add, a, fields[1]);
        ret(&mut main, result);
        let id = main.id;
        let mut module = HirModule::new(InternedString::new_global("wide_unions"));
        for f in [make, main] {
            module.functions.insert(f.id, f);
        }
        check_all(
            &module,
            id,
            (width as f64 + 2.0) * 100.0 + width as f64 + 4.0,
        );
    }
}

#[test]
fn loop_phis_capture_values_before_constructor_and_phi_slots_are_reused() {
    let ty = union(HirType::U8, HirType::F64, false);
    let mut f = function("loopScore", vec![], vec![HirType::F64]);
    let entry = f.entry_block;
    let head = HirId::new();
    let body = HirId::new();
    let exit = HirId::new();
    let mut seeds = Vec::new();
    for n in [0.0, 7.0, 9.0] {
        let payload = constant(&mut f, n);
        let result = value(&mut f, ty.clone(), HirValueKind::Instruction);
        emit(
            &mut f,
            HirInstruction::CreateUnion {
                result,
                union_ty: ty.clone(),
                variant_index: 1,
                value: payload,
            },
        );
        seeds.push(result);
    }
    let zero = constant(&mut f, 0.0);
    let one = constant(&mut f, 1.0);
    let four = constant(&mut f, 4.0);
    let prev = value(&mut f, ty.clone(), HirValueKind::Instruction);
    let a = value(&mut f, ty.clone(), HirValueKind::Instruction);
    let b = value(&mut f, ty.clone(), HirValueKind::Instruction);
    let i = value(&mut f, HirType::F64, HirValueKind::Instruction);
    let sum = value(&mut f, HirType::F64, HirValueKind::Instruction);
    let new_i = value(&mut f, HirType::F64, HirValueKind::Instruction);
    let new_sum = value(&mut f, HirType::F64, HirValueKind::Instruction);
    let fresh = value(&mut f, ty.clone(), HirValueKind::Instruction);
    let old = value(&mut f, HirType::F64, HirValueKind::Instruction);
    let cond = value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let e = f.blocks.get_mut(&entry).unwrap();
    e.terminator = HirTerminator::Branch { target: head };
    e.successors = vec![head];
    let mut h = HirBlock::new(head);
    h.predecessors = vec![entry, body];
    h.successors = vec![body, exit];
    for (result, ty, seed, next) in [
        (prev, ty.clone(), seeds[0], fresh),
        (a, ty.clone(), seeds[1], b),
        (b, ty.clone(), seeds[2], a),
        (i, HirType::F64, zero, new_i),
        (sum, HirType::F64, zero, new_sum),
    ] {
        h.phis.push(HirPhi {
            result,
            ty,
            incoming: vec![(seed, entry), (next, body)],
        });
    }
    h.instructions.push(HirInstruction::Binary {
        op: BinaryOp::FLt,
        result: cond,
        ty: HirType::F64,
        left: i,
        right: four,
    });
    h.terminator = HirTerminator::CondBranch {
        condition: cond,
        true_target: body,
        false_target: exit,
    };
    let mut block = HirBlock::new(body);
    block.predecessors = vec![head];
    block.successors = vec![head];
    block.instructions = vec![
        HirInstruction::Binary {
            op: BinaryOp::Add,
            result: new_i,
            ty: HirType::F64,
            left: i,
            right: one,
        },
        HirInstruction::CreateUnion {
            result: fresh,
            union_ty: ty,
            variant_index: 1,
            value: new_i,
        },
        HirInstruction::ExtractUnionValue {
            result: old,
            ty: HirType::F64,
            union_val: prev,
            variant_index: 1,
        },
        HirInstruction::Binary {
            op: BinaryOp::Add,
            result: new_sum,
            ty: HirType::F64,
            left: sum,
            right: old,
        },
    ];
    block.terminator = HirTerminator::Branch { target: head };
    let mut done = HirBlock::new(exit);
    done.predecessors = vec![head];
    let saved_entry = f.entry_block;
    f.entry_block = exit;
    f.blocks.insert(exit, done);
    let av = payload(&mut f, a);
    let bv = payload(&mut f, b);
    let ten = constant(&mut f, 10.0);
    let av = binary(&mut f, BinaryOp::Mul, av, ten);
    let result = binary(&mut f, BinaryOp::Add, sum, av);
    let result = binary(&mut f, BinaryOp::Add, result, bv);
    ret(&mut f, result);
    f.entry_block = saved_entry;
    f.blocks.insert(head, h);
    f.blocks.insert(body, block);
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("union_loop"));
    module.functions.insert(id, f);
    check_all(&module, id, 85.0);
}

#[test]
fn layout_includes_payload_alignment_and_nested_padding() {
    use zyntax_compiler::value_layout::{size_of_hir_ty, union_layout};
    let nested = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::U8, HirType::F64, HirType::U8].into(),
        packed: false,
    });
    assert_eq!(size_of_hir_ty(&nested), 24);
    assert_eq!(
        zyntax_compiler::object_header::field_layout(&[
            HirType::I64,
            union(HirType::U8, HirType::F64, false)
        ]),
        (vec![0, 8], 24),
    );
    for (ty, offset, size) in [
        (union(HirType::U8, HirType::F64, false), 8, 16),
        (union(HirType::U64, HirType::Bool, false), 8, 16),
        (union(HirType::U8, nested, false), 8, 32),
        (union(HirType::U32, HirType::F64, true), 0, 8),
    ] {
        let HirType::Union(u) = ty else {
            unreachable!()
        };
        let layout = union_layout(&u);
        assert_eq!(layout.payload_offset, offset);
        assert_eq!(layout.size, size);
        assert_eq!(layout.align, 8);
    }
}

#[test]
fn interpreter_owns_returned_values_and_load_copies() {
    for (tag, c_union) in [
        (HirType::U8, false),
        (HirType::U32, false),
        (HirType::U64, false),
        (HirType::U32, true),
    ] {
        let (module, ..) = module(tag, c_union);
        let mut interp = zyntax_compiler::hir_interp::HirInterpreter::new();
        assert_eq!(
            interp.call(&module, "score", vec![]).unwrap(),
            zyntax_compiler::value::ZyntaxValue::Float(if c_union { 1766.5 } else { 1775.5 })
        );
    }
}

#[cfg(feature = "cranelift-backend")]
#[test]
fn cranelift_owns_returned_values_and_load_copies() {
    for (tag, c_union) in [
        (HirType::U8, false),
        (HirType::U32, false),
        (HirType::U64, false),
        (HirType::U32, true),
    ] {
        let (module, main, ..) = module(tag, c_union);
        let mut backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
        backend.compile_module(&module).unwrap();
        backend.finalize_definitions().unwrap();
        let run: unsafe extern "C" fn() -> f64 =
            unsafe { std::mem::transmute(backend.get_function_ptr(main).unwrap()) };
        assert_eq!(unsafe { run() }, if c_union { 1766.5 } else { 1775.5 });
    }
}

#[cfg(feature = "llvm-backend")]
#[test]
fn the_default_llvm_engine_preserves_union_return_storage() {
    let (module, main, ..) = module(HirType::U8, false);
    let context = inkwell::context::Context::create();
    let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
    llvm.compile_module(&module).unwrap();
    let run: unsafe extern "C" fn() -> f64 =
        unsafe { std::mem::transmute(llvm.get_function_pointer(main).unwrap()) };
    assert_eq!(unsafe { run() }, 1775.5);
}

#[cfg(feature = "cranelift-backend")]
#[test]
fn the_interpreter_native_bridge_passes_union_arguments_and_return_destinations() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let (module, _, make, echo) = module(HirType::U8, false);
    let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    clif.compile_module(&module).unwrap();
    clif.finalize_definitions().unwrap();
    let make_ptr = clif.get_function_ptr(make).unwrap() as usize;
    let echo_ptr = clif.get_function_ptr(echo).unwrap() as usize;
    let make_sig =
        zyntax_compiler::hir_interp::NativeSig::of_function(&module.functions[&make], false);
    let echo_sig =
        zyntax_compiler::hir_interp::NativeSig::of_function(&module.functions[&echo], false);
    let make_thunk = clif.interp_thunk(&make_sig).unwrap() as usize;
    let echo_thunk = clif.interp_thunk(&echo_sig).unwrap() as usize;
    let entries = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
    let native_hits = Arc::new(AtomicUsize::new(0));
    let read_entries = Arc::clone(&entries);
    let read_hits = Arc::clone(&native_hits);
    let mut interp = zyntax_compiler::hir_interp::HirInterpreter::new();
    interp.set_native_bridge(
        Box::new(move |sig| {
            if *sig == make_sig {
                Some(make_thunk as *const u8)
            } else if *sig == echo_sig {
                Some(echo_thunk as *const u8)
            } else {
                None
            }
        }),
        Box::new(move |id| {
            let index = if id == make {
                0
            } else if id == echo {
                1
            } else {
                return None;
            };
            let ptr = read_entries[index].load(Ordering::Acquire);
            if ptr == 0 {
                None
            } else {
                read_hits.fetch_add(1, Ordering::Relaxed);
                Some(ptr as *const u8)
            }
        }),
        Box::new(|_| None),
    );
    for (id, index) in [(make, 0), (echo, 1)] {
        let entries = Arc::clone(&entries);
        let hits = Arc::clone(&native_hits);
        interp.register_tick_callback(
            id,
            Box::new(move || {
                let ptr = entries[index].load(Ordering::Acquire);
                if ptr == 0 {
                    None
                } else {
                    hits.fetch_add(1, Ordering::Relaxed);
                    Some(ptr as *const u8)
                }
            }),
        );
    }
    let check = |interp: &mut zyntax_compiler::hir_interp::HirInterpreter| {
        assert_eq!(
            interp.call(&module, "score", vec![]).unwrap(),
            zyntax_compiler::value::ZyntaxValue::Float(1775.5)
        );
    };
    check(&mut interp);
    assert_eq!(native_hits.load(Ordering::Relaxed), 0);
    entries[0].store(make_ptr, Ordering::Release);
    check(&mut interp);
    let hits = native_hits.load(Ordering::Relaxed);
    assert!(hits >= 2, "the constructor ran in native code");
    entries[1].store(echo_ptr, Ordering::Release);
    check(&mut interp);
    assert!(
        native_hits.load(Ordering::Relaxed) >= hits + 3,
        "both native union signatures were entered"
    );
    interp.clear_native_bridge();
}

#[cfg(feature = "llvm-backend")]
#[test]
fn llvm_and_mixed_native_calls_keep_union_values() {
    use inkwell::context::Context;
    use std::sync::Arc;
    use zyntax_compiler::{cranelift_backend::CraneliftBackend, llvm_jit_backend::LLVMJitBackend};
    for (tag, c_union) in [
        (HirType::U8, false),
        (HirType::U32, false),
        (HirType::U64, false),
        (HirType::U32, true),
    ] {
        let (module, main, make, echo) = module(tag, c_union);
        let expected = if c_union { 1766.5 } else { 1775.5 };
        let mut clif = CraneliftBackend::new().unwrap();
        clif.set_reloadable_calls(true);
        clif.compile_module(&module).unwrap();
        clif.finalize_definitions().unwrap();
        let run: unsafe extern "C" fn() -> f64 =
            unsafe { std::mem::transmute(clif.get_function_ptr(main).unwrap()) };
        let context = Context::create();
        let mut llvm = LLVMJitBackend::new(&context).unwrap();
        llvm.set_use_mcjit(true);
        llvm.set_module_context(Arc::new(module.clone()));
        llvm.set_cross_tier_links(clif.reload_key(), Arc::new(|_| None));
        llvm.compile_function(make, &module.functions[&make])
            .unwrap();
        clif.publish_call_target(make, llvm.get_function_pointer(make).unwrap() as usize);
        assert_eq!(
            unsafe { run() },
            expected,
            "Cranelift caller / LLVM constructor"
        );
        llvm.compile_function(main, &module.functions[&main])
            .unwrap();
        let native: unsafe extern "C" fn() -> f64 =
            unsafe { std::mem::transmute(llvm.get_function_pointer(main).unwrap()) };
        assert_eq!(
            unsafe { native() },
            expected,
            "LLVM caller / Cranelift echo"
        );
        llvm.compile_function(echo, &module.functions[&echo])
            .unwrap();
        clif.publish_call_target(echo, llvm.get_function_pointer(echo).unwrap() as usize);
        assert_eq!(unsafe { native() }, expected, "LLVM caller / LLVM echo");
    }
}
