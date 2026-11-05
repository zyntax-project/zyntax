use super::*;

#[test]
fn indexed_alias_uses_preserve_block_order_and_indirect_callees() {
    let (mut f, ptr) = build_alloc_store_load_return();
    let entry = f.entry_block;
    let other = f.create_block();
    let ty = f.values[&ptr].ty.clone();
    let alias = add_inst_val(&mut f, ty.clone());
    f.blocks[&entry].instructions.push(HirInstruction::Cast {
        result: alias,
        op: crate::hir::CastOp::Bitcast,
        operand: ptr,
        ty,
    });
    f.blocks[&other].instructions = vec![
        HirInstruction::Store {
            value: ptr,
            ptr: alias,
            align: 8,
            volatile: false,
        },
        HirInstruction::Call {
            result: None,
            callee: HirCallable::Indirect(alias),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
    ];
    // Block order need not agree with identifier order or CFG order.
    f.blocks.swap_indices(0, 1);
    let values = IdSet::from_iter([ptr, alias]);
    let users = Users::of(&f);
    let indexed = users.instruction_uses(&f, &values);
    assert_eq!(
        indexed.as_slice(),
        &[(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]
    );
    let mut scanned = Vec::new();
    for (b, block) in f.blocks.values().enumerate() {
        for (i, inst) in block.instructions.iter().enumerate() {
            if inst.any_operand(|v| values.contains(&v)) {
                scanned.push((b, i));
            }
        }
    }
    assert_eq!(indexed.as_slice(), scanned);
}

#[test]
fn inserting_a_release_reindexes_later_sites() {
    let (mut f, ptr) = build_alloc_store_load_return();
    let entry = f.entry_block;
    let ty = f.values[&ptr].ty.clone();
    let second = add_inst_val(&mut f, ty);
    let loaded = add_inst_val(&mut f, HirType::I64);
    let mut insts = f.blocks[&entry].instructions.clone();
    if let HirInstruction::Call { result, .. } = &mut insts[0] {
        *result = Some(second);
    }
    if let HirInstruction::Store { ptr, .. } = &mut insts[1] {
        *ptr = second;
    }
    if let HirInstruction::Load { ptr, result, .. } = &mut insts[2] {
        *ptr = second;
        *result = loaded;
    }
    f.blocks[&entry].instructions.extend(insts);
    f.blocks[&entry].terminator = HirTerminator::Return {
        values: vec![loaded],
    };
    let stats = run_function(&mut f, &ModuleFacts::default());
    assert_eq!(stats.frees_inserted, 2);
    let insts = &f.blocks[&entry].instructions;
    for (index, ptr) in [(3, ptr), (7, second)] {
        assert!(matches!(&insts[index], HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Free), args, ..
        } if args == &[ptr]));
    }
    let original = bincode::serialize(&f).unwrap();
    assert_eq!(
        run_function(&mut f, &ModuleFacts::default()).frees_inserted,
        0
    );
    assert_eq!(bincode::serialize(&f).unwrap(), original);
}

#[test]
fn many_escaping_allocations_keep_their_storage() {
    let mut f = HirFunction::new(
        InternedString::new_global("many_escapes"),
        empty_sig(HirType::Void),
    );
    let entry = f.entry_block;
    let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
    for _ in 0..2048 {
        let result = add_inst_val(&mut f, HirType::I64);
        f.blocks[&entry].instructions.push(HirInstruction::Binary {
            result,
            op: crate::hir::BinaryOp::Add,
            left: size,
            right: size,
            ty: HirType::I64,
        });
    }
    for _ in 0..256 {
        let ptr = add_inst_val(&mut f, HirType::Ptr(std::sync::Arc::new(HirType::I64)));
        for (result, callee, args) in [
            (
                Some(ptr),
                HirCallable::Intrinsic(Intrinsic::Malloc),
                vec![size],
            ),
            (
                None,
                HirCallable::Symbol("unknown_consumer".into()),
                vec![ptr],
            ),
        ] {
            f.blocks[&entry].instructions.push(HirInstruction::Call {
                result,
                callee,
                args,
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        }
    }
    f.blocks[&entry].terminator = HirTerminator::Return { values: vec![] };
    let original = bincode::serialize(&f).unwrap();
    let start = std::time::Instant::now();
    let stats = run_function(&mut f, &ModuleFacts::default());
    eprintln!("escaping allocation analysis: {:?}", start.elapsed());
    assert_eq!(stats.mallocs_scanned, 256);
    assert_eq!(stats.escapes_skipped, 256);
    assert_eq!(stats.frees_inserted, 0);
    assert_eq!(bincode::serialize(&f).unwrap(), original);
}
