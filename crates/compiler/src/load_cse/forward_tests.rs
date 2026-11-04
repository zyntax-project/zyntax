use super::*;
use crate::hir::{AtomicOrdering, HirCallable, HirFunctionSignature, HirTerminator};
use std::sync::Arc;

fn fixture() -> (HirFunction, HirId, HirId) {
    let mut f = HirFunction::new(
        InternedString::new_global("forward"),
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
    let ptr = f.create_value(
        HirType::Ptr(Arc::new(HirType::I64)),
        HirValueKind::Parameter(0),
    );
    let value = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(37)));
    store(&mut f, ptr, value, false);
    (f, ptr, value)
}
fn store(f: &mut HirFunction, ptr: HirId, value: HirId, volatile: bool) {
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .push(HirInstruction::Store {
            value,
            ptr,
            align: 1,
            volatile,
        });
}
fn load(f: &mut HirFunction, ptr: HirId, ty: HirType, volatile: bool) -> HirId {
    let result = f.create_value(ty.clone(), HirValueKind::Instruction);
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .push(HirInstruction::Load {
            result,
            ty,
            ptr,
            align: 1,
            volatile,
        });
    f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::Return {
        values: vec![result],
    };
    result
}

#[test]
fn typed_store_is_forwarded_but_same_width_reinterpretation_is_not() {
    for ty in [HirType::I64, HirType::U64, HirType::F64, HirType::I8] {
        let (mut f, ptr, value) = fixture();
        load(&mut f, ptr, ty.clone(), false);
        assert_eq!(
            run_regions(&mut f).eliminated,
            usize::from(ty == HirType::I64)
        );
        if ty == HirType::I64 {
            assert!(matches!(&f.blocks[&f.entry_block].terminator,
                HirTerminator::Return { values } if values == &[value]));
        }
    }
}

#[test]
fn calls_fences_and_volatile_accesses_discard_stored_values() {
    for variant in 0..4 {
        let (mut f, ptr, value) = fixture();
        match variant {
            0 => store(&mut f, ptr, value, true),
            1 => {
                load(&mut f, ptr, HirType::I64, true);
            }
            2 => {
                f.blocks
                    .get_mut(&f.entry_block)
                    .unwrap()
                    .instructions
                    .push(HirInstruction::Fence {
                        ordering: AtomicOrdering::SeqCst,
                    })
            }
            _ => {
                f.blocks
                    .get_mut(&f.entry_block)
                    .unwrap()
                    .instructions
                    .push(HirInstruction::Call {
                        result: None,
                        callee: HirCallable::Symbol("mutate".into()),
                        args: vec![ptr],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    })
            }
        }
        load(&mut f, ptr, HirType::I64, false);
        assert_eq!(run_regions(&mut f).eliminated, 0, "barrier {variant}");
    }
}

#[test]
fn unknown_alias_and_partial_write_kill_the_available_store() {
    for alias in [false, true] {
        let (mut f, ptr, _) = fixture();
        let q = if alias {
            f.create_value(
                HirType::Ptr(Arc::new(HirType::I64)),
                HirValueKind::Parameter(1),
            )
        } else {
            ptr
        };
        let byte = f.create_value(HirType::I8, HirValueKind::Constant(HirConstant::I8(3)));
        store(&mut f, q, byte, false);
        load(&mut f, ptr, HirType::I64, false);
        assert_eq!(run_regions(&mut f).eliminated, 0);
    }
}

#[test]
fn separate_ssa_values_for_one_global_still_alias() {
    let (mut f, ptr, value) = fixture();
    let global = HirId::new();
    f.values.get_mut(&ptr).unwrap().kind = HirValueKind::Global(global);
    let q = f.create_value(
        HirType::Ptr(Arc::new(HirType::I64)),
        HirValueKind::Global(global),
    );
    let replacement = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(9)));
    store(&mut f, q, replacement, false);
    let loaded = load(&mut f, ptr, HirType::I64, false);
    assert_eq!(run_regions(&mut f).eliminated, 0);
    assert_ne!(loaded, value);
}

#[test]
fn global_store_cannot_overwrite_a_fresh_allocation() {
    for fresh in [false, true] {
        let (mut f, ptr, _) = fixture();
        if fresh {
            f.values.get_mut(&ptr).unwrap().kind = HirValueKind::Instruction;
            f.blocks
                .get_mut(&f.entry_block)
                .unwrap()
                .instructions
                .insert(
                    0,
                    HirInstruction::Alloca {
                        result: ptr,
                        ty: HirType::I64,
                        count: None,
                        align: 8,
                    },
                );
        }
        let g = f.create_value(
            HirType::Ptr(Arc::new(HirType::I64)),
            HirValueKind::Global(HirId::new()),
        );
        let line = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(44)));
        store(&mut f, g, line, false);
        load(&mut f, ptr, HirType::I64, false);
        assert_eq!(run_regions(&mut f).eliminated, usize::from(fresh));
    }
}

#[test]
fn pointer_stores_are_not_forwarded_as_aggregate_carriers() {
    let (mut f, ptr, _) = fixture();
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .clear();
    store(&mut f, ptr, ptr, false);
    load(&mut f, ptr, HirType::Ptr(Arc::new(HirType::I64)), false);
    assert_eq!(run_regions(&mut f).eliminated, 0);
}

#[test]
fn forwarded_reads_allow_overwrites_but_barriers_and_partial_writes_do_not() {
    for variant in 0..4 {
        let (mut f, ptr, value) = fixture();
        load(&mut f, ptr, HirType::I64, false);
        match variant {
            1 => {
                load(&mut f, ptr, HirType::I64, true);
            }
            2 => {
                f.blocks
                    .get_mut(&f.entry_block)
                    .unwrap()
                    .instructions
                    .push(HirInstruction::Fence {
                        ordering: AtomicOrdering::SeqCst,
                    })
            }
            _ => {}
        }
        let replacement = if variant == 3 {
            f.create_value(HirType::I8, HirValueKind::Constant(HirConstant::I8(9)))
        } else {
            value
        };
        store(&mut f, ptr, replacement, false);
        run_regions(&mut f);
        let stores = f.blocks[&f.entry_block]
            .instructions
            .iter()
            .filter(|i| matches!(i, HirInstruction::Store { .. }))
            .count();
        assert_eq!(stores, if variant == 0 { 1 } else { 2 });
    }
}

#[test]
fn global_identity_and_fresh_fields_allow_independent_dead_stores() {
    let (mut f, ptr, value) = fixture();
    f.values.get_mut(&ptr).unwrap().kind = HirValueKind::Instruction;
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .insert(
            0,
            HirInstruction::Alloca {
                result: ptr,
                ty: HirType::I64,
                count: None,
                align: 8,
            },
        );
    let id = HirId::new();
    let g1 = f.create_value(
        HirType::Ptr(Arc::new(HirType::I64)),
        HirValueKind::Global(id),
    );
    let g2 = f.create_value(
        HirType::Ptr(Arc::new(HirType::I64)),
        HirValueKind::Global(id),
    );
    store(&mut f, g1, value, false);
    load(&mut f, ptr, HirType::I64, false);
    store(&mut f, ptr, value, false);
    store(&mut f, g2, value, false);
    run_regions(&mut f);
    let pointers: Vec<_> = f.blocks[&f.entry_block]
        .instructions
        .iter()
        .filter_map(|i| match i {
            HirInstruction::Store { ptr, .. } => Some(*ptr),
            _ => None,
        })
        .collect();
    assert_eq!(pointers, [ptr, g2]);
}
