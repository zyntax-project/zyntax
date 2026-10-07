//! Byte-range proofs preserve overlapping writes and aliases in every tier.
#![cfg(feature = "cranelift-backend")]
mod common;

use zyntax_compiler::{hir::*, hir_interp::HirInterpreter, load_cse, value::ZyntaxValue};
use zyntax_typed_ast::InternedString;

fn fixture(
    load_offset: i64,
    store_offset: i64,
    separate_root: bool,
    scaled: bool,
) -> (HirModule, HirId) {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![HirType::I64];
    sig.is_pure = false;
    let mut f = HirFunction::new(InternedString::new_global("ranges"), sig);
    for i in 0..2 {
        let ty = HirType::Ptr(Box::new(HirType::I64));
        let id = f.create_value(ty.clone(), HirValueKind::Parameter(i));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global("ptr"),
            ty,
            attributes: Default::default(),
            ownership: Default::default(),
        });
    }
    let read_base = f.signature.params[0].id;
    let write_base = f.signature.params[usize::from(separate_root)].id;
    let index_ty = if scaled {
        HirType::Ptr(Box::new(HirType::I64))
    } else {
        HirType::U8
    };
    let mut gep = |base, offset| {
        let index = f.create_value(
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(offset)),
        );
        let result = f.create_value(
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        f.blocks.get_mut(&f.entry_block).unwrap().instructions.push(
            HirInstruction::GetElementPtr {
                result,
                ty: index_ty.clone(),
                ptr: base,
                indices: vec![index],
            },
        );
        result
    };
    let p = gep(read_base, load_offset);
    let q = gep(write_base, store_offset);
    // Exercise the pointer casts used around byte GEPs in lowered code.
    let word = f.create_value(HirType::U64, HirValueKind::Instruction);
    let roundtrip = f.create_value(
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let cast_q = f.create_value(
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let first = f.create_value(HirType::I64, HirValueKind::Instruction);
    let second = f.create_value(HirType::I64, HirValueKind::Instruction);
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    let value = f.create_value(
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(0x1122334455667788)),
    );
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.instructions.extend([
        HirInstruction::Cast {
            result: word,
            op: CastOp::PtrToInt,
            ty: HirType::U64,
            operand: q,
        },
        HirInstruction::Cast {
            result: roundtrip,
            op: CastOp::IntToPtr,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            operand: word,
        },
        HirInstruction::Cast {
            result: cast_q,
            op: CastOp::Bitcast,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            operand: roundtrip,
        },
        HirInstruction::Load {
            result: first,
            ty: HirType::I64,
            ptr: p,
            align: 1,
            volatile: false,
        },
        HirInstruction::Store {
            value,
            ptr: cast_q,
            align: 1,
            volatile: false,
        },
        HirInstruction::Load {
            result: second,
            ty: HirType::I64,
            ptr: p,
            align: 1,
            volatile: false,
        },
        HirInstruction::Binary {
            result,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: first,
            right: second,
        },
    ]);
    entry.terminator = HirTerminator::Return {
        values: vec![result],
    };
    let id = f.id;
    let mut m = HirModule::new(InternedString::new_global("ranges"));
    m.functions.insert(id, f);
    (m, id)
}

fn check(mut call: impl FnMut(*mut u8, *mut u8) -> i64, load_offset: i64, store_offset: i64) {
    #[repr(align(16))]
    struct Buffer([u8; 64]);
    for seed in [0u8, 1, 0xff] {
        let mut buffer = Buffer([seed; 64]);
        let bytes = &mut buffer.0;
        let read = (16 + load_offset) as usize;
        let write = (16 + store_offset) as usize;
        let first = i64::from_ne_bytes(bytes[read..read + 8].try_into().unwrap());
        let mut expected = *bytes;
        expected[write..write + 8].copy_from_slice(&0x1122334455667788i64.to_ne_bytes());
        let second = i64::from_ne_bytes(expected[read..read + 8].try_into().unwrap());
        // SAFETY: every fixture access is within this buffer; unaligned access is explicit.
        let base = unsafe { bytes.as_mut_ptr().add(16) };
        assert_eq!(call(base, base), first.wrapping_add(second));
        assert_eq!(*bytes, expected);
    }
}

#[test]
fn disjoint_fields_and_overlapping_aliases_execute_in_every_tier() {
    for (read, write, scaled, distinct, folded) in [
        (0, 8, false, false, 1),
        (8, 0, false, false, 1),
        (0, 4, false, false, 0),
        (4, 0, false, false, 0),
        (0, 0, false, false, 0),
        (0, -8, false, false, 0), // The range endpoint wraps: stay conservative.
        (-8, 0, false, false, 0),
        (0, -4, false, false, 0),
        (0, 1, true, false, 1),
        (0, 1, false, false, 0),
        (0, 0, false, true, 0),
        (0, 8, false, true, 0),
    ] {
        for optimize in [false, true] {
            let (mut module, id) = fixture(read, write, distinct, scaled);
            if optimize {
                assert_eq!(
                    load_cse::run(module.functions.get_mut(&id).unwrap()).eliminated,
                    folded,
                    "read={read} write={write} scaled={scaled} distinct={distinct}"
                );
            }
            let stride = if scaled { 8 } else { 1 };
            // The interpreter's raw scalar reads require natural alignment.
            if (read * stride) % 8 == 0 && (write * stride) % 8 == 0 {
                let mut interp = HirInterpreter::new();
                check(
                    |p, q| {
                        let result = interp
                            .call(
                                &module,
                                "ranges",
                                vec![ZyntaxValue::Pointer(p), ZyntaxValue::Pointer(q)],
                            )
                            .unwrap();
                        match result {
                            ZyntaxValue::Int(n) => n,
                            other => panic!("{other:?}"),
                        }
                    },
                    read * stride,
                    write * stride,
                );
            }
            let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
            clif.compile_module(&module).unwrap();
            clif.finalize_definitions().unwrap();
            // SAFETY: fixture signature is C (*mut u8, *mut u8) -> i64; the JIT stays alive.
            let call: unsafe extern "C" fn(*mut u8, *mut u8) -> i64 =
                unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
            check(|p, q| unsafe { call(p, q) }, read * stride, write * stride);
            #[cfg(feature = "llvm-backend")]
            {
                let context = inkwell::context::Context::create();
                let mut llvm =
                    zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
                llvm.compile_module(&module).unwrap();
                // SAFETY: same fixture signature, with the backend kept alive.
                let call: unsafe extern "C" fn(*mut u8, *mut u8) -> i64 =
                    unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
                check(|p, q| unsafe { call(p, q) }, read * stride, write * stride);
            }
        }
    }
}

#[test]
fn volatile_and_differently_typed_loads_are_not_reused() {
    for variant in 0..3 {
        let (mut module, id) = fixture(0, 8, false, false);
        let f = module.functions.get_mut(&id).unwrap();
        let mut loads = 0;
        for inst in &mut f.blocks.get_mut(&f.entry_block).unwrap().instructions {
            match inst {
                HirInstruction::Load {
                    result,
                    ty,
                    volatile,
                    ..
                } => {
                    loads += 1;
                    if variant == 0 {
                        *volatile = true;
                    }
                    if variant == 2 && loads == 2 {
                        *ty = HirType::I32;
                        f.values.get_mut(result).unwrap().ty = HirType::I32;
                    }
                }
                HirInstruction::Store { volatile, .. } if variant == 1 => *volatile = true,
                _ => {}
            }
        }
        assert_eq!(load_cse::run(f).eliminated, 0);
    }
}

#[test]
fn dead_predecessors_expose_disjoint_fields_to_the_pipeline() {
    let (mut module, id) = fixture(0, 8, false, false);
    let f = module.functions.get_mut(&id).unwrap();
    let entry_id = f.entry_block;
    let rest_id = HirId::new();
    let dead_id = HirId::new();
    let entry = f.blocks.get_mut(&entry_id).unwrap();
    let store = entry
        .instructions
        .iter()
        .position(|i| matches!(i, HirInstruction::Store { .. }))
        .unwrap();
    let mut rest = HirBlock::new(rest_id);
    rest.instructions = entry.instructions.split_off(store);
    rest.terminator = entry.terminator.clone();
    rest.predecessors = vec![entry_id, dead_id];
    entry.terminator = HirTerminator::Branch { target: rest_id };
    entry.successors = vec![rest_id];
    let mut dead = HirBlock::new(dead_id);
    dead.terminator = HirTerminator::Branch { target: rest_id };
    dead.successors = vec![rest_id];
    f.blocks.insert(rest_id, rest);
    f.blocks.insert(dead_id, dead);
    let stats = zyntax_compiler::run_interp_safe_opts(&mut module);
    assert!(stats.cfg_simplify.unreachable_removed > 0);
    assert_eq!(module.functions[&id].blocks.len(), 1);
    assert_eq!(
        module.functions[&id]
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(i, HirInstruction::Load { .. }))
            .count(),
        1
    );
    let mut interp = HirInterpreter::new();
    check(
        |p, q| match interp
            .call(
                &module,
                "ranges",
                vec![ZyntaxValue::Pointer(p), ZyntaxValue::Pointer(q)],
            )
            .unwrap()
        {
            ZyntaxValue::Int(n) => n,
            other => panic!("{other:?}"),
        },
        0,
        8,
    );
}

#[test]
fn array_layouts_and_multi_index_geps_do_not_supply_range_proofs() {
    for multi_index in [false, true] {
        let (mut module, id) = fixture(16, 1, false, false);
        let f = module.functions.get_mut(&id).unwrap();
        let geps = f
            .blocks
            .get_mut(&f.entry_block)
            .unwrap()
            .instructions
            .iter_mut()
            .filter(|i| matches!(i, HirInstruction::GetElementPtr { .. }));
        if let Some(HirInstruction::GetElementPtr { ty, indices, .. }) = geps.skip(1).next() {
            if multi_index {
                indices.push(indices[0]);
            } else {
                *ty = HirType::Array(Box::new(HirType::I64), 2);
            }
        }
        assert_eq!(load_cse::run(f).eliminated, 0);
    }
}

#[test]
fn aggregate_store_carriers_do_not_understate_the_written_range() {
    let (mut module, id) = fixture(8, 0, false, false);
    let f = module.functions.get_mut(&id).unwrap();
    let ty = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::I64; 3],
        packed: false,
    });
    let aggregate = f.create_value(
        HirType::Ptr(Box::new(ty.clone())),
        HirValueKind::Instruction,
    );
    let block = f.blocks.get_mut(&f.entry_block).unwrap();
    block.instructions.insert(
        0,
        HirInstruction::Load {
            result: aggregate,
            ty,
            ptr: f.signature.params[1].id,
            align: 8,
            volatile: false,
        },
    );
    for inst in &mut block.instructions {
        if let HirInstruction::Store { value, .. } = inst {
            *value = aggregate;
        }
    }
    assert_eq!(load_cse::run(f).eliminated, 0);
}

#[test]
fn late_block_merging_splits_aggregate_reads_into_fields() {
    let (mut module, id) = fixture(0, 8, false, false);
    let f = module.functions.get_mut(&id).unwrap();
    let ptr = f.signature.params[0].id;
    let ty = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::I64; 3],
        packed: false,
    });
    let aggregate = f.create_value(ty.clone(), HirValueKind::Instruction);
    let field = f.create_value(HirType::I64, HirValueKind::Instruction);
    let entry_id = f.entry_block;
    let rest_id = HirId::new();
    let dead_id = HirId::new();
    let entry = f.blocks.get_mut(&entry_id).unwrap();
    entry.instructions = vec![HirInstruction::Load {
        result: aggregate,
        ty,
        ptr,
        align: 8,
        volatile: false,
    }];
    entry.terminator = HirTerminator::Branch { target: rest_id };
    entry.successors = vec![rest_id];
    let mut rest = HirBlock::new(rest_id);
    rest.predecessors = vec![entry_id, dead_id];
    rest.instructions.push(HirInstruction::ExtractValue {
        result: field,
        ty: HirType::I64,
        aggregate,
        indices: vec![1],
    });
    rest.terminator = HirTerminator::Return {
        values: vec![field],
    };
    let mut dead = HirBlock::new(dead_id);
    dead.terminator = HirTerminator::Branch { target: rest_id };
    dead.successors = vec![rest_id];
    f.blocks.insert(rest_id, rest);
    f.blocks.insert(dead_id, dead);

    let stats = zyntax_compiler::run_interp_safe_opts(&mut module);
    assert!(stats.cfg_simplify.unreachable_removed > 0);
    assert_eq!(stats.aggregate_split.field_reads_only, 1);
    let f = &module.functions[&id];
    assert_eq!(f.blocks.len(), 1);
    assert!(!f.blocks.values().flat_map(|b| &b.instructions).any(|i| {
        matches!(
            i,
            HirInstruction::Load {
                ty: HirType::Struct(_),
                ..
            } | HirInstruction::ExtractValue { .. }
        )
    }));
    let mut memory = [11i64, 29, 47];
    let p = memory.as_mut_ptr().cast();
    let mut interp = HirInterpreter::new();
    assert!(matches!(
        interp
            .call(&module, "ranges", vec![ZyntaxValue::Pointer(p); 2])
            .unwrap(),
        ZyntaxValue::Int(29)
    ));
}
