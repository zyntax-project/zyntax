#![cfg(feature = "cranelift-backend")]

mod common;
use zyntax_compiler::{hir::*, osr, sign_fold};
use zyntax_typed_ast::InternedString;

fn pointer() -> HirType {
    HirType::Ptr(std::sync::Arc::new(HirType::U8))
}

// Keep both an allocation and a possibly-null parameter live across the header.
fn fixture(nullable_backedge: bool) -> (HirFunction, HirId, HirId, HirId, HirId) {
    let (mut f, header) = common::counted_loop();
    let input = f.create_value(pointer(), HirValueKind::Parameter(1));
    f.signature.params.push(HirParam {
        id: input,
        name: InternedString::new_global("input"),
        ty: pointer(),
        attributes: Default::default(),
        ownership: Default::default(),
    });
    let made = f.create_value(pointer(), HirValueKind::Instruction);
    let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
    let null = f.create_value(
        pointer(),
        HirValueKind::Constant(HirConstant::Null(pointer())),
    );
    let current = f.create_value(pointer(), HirValueKind::Instruction);
    let checked = f.create_value(HirType::Bool, HirValueKind::Instruction);
    let checked_input = f.create_value(HirType::Bool, HirValueKind::Instruction);
    let entry = f.entry_block;
    let latch = f.blocks[&header]
        .predecessors
        .iter()
        .copied()
        .find(|b| *b != entry)
        .unwrap();
    f.blocks
        .get_mut(&entry)
        .unwrap()
        .instructions
        .push(HirInstruction::Call {
            result: Some(made),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
    f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
        result: current,
        ty: pointer(),
        incoming: vec![
            (made, entry),
            (if nullable_backedge { input } else { current }, latch),
        ],
    });
    for (result, left) in [(checked, current), (checked_input, input)] {
        f.blocks
            .get_mut(&header)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                result,
                left,
                right: null,
                op: BinaryOp::Ne,
                ty: HirType::Bool,
            });
    }
    (f, header, current, checked, checked_input)
}

#[test]
fn allocation_fact_survives_a_loop_phi_and_outlining_without_noalias() {
    let (f, header, current, checked, checked_input) = fixture(false);
    let layout = osr::osr_layout(&f, header).unwrap();
    let index = layout
        .live_ins
        .iter()
        .position(|id| *id == current)
        .unwrap();
    let outlined = osr::outline(
        &f,
        &layout,
        HirId::new(),
        InternedString::new_global("region"),
    )
    .unwrap();
    assert!(outlined.function.signature.params[index].attributes.nonnull);
    assert!(
        outlined
            .function
            .signature
            .params
            .iter()
            .all(|p| !p.attributes.noalias)
    );
    let mut module = HirModule::new(InternedString::new_global("facts"));
    let id = outlined.function.id;
    module.functions.insert(id, outlined.function);
    sign_fold::run_module(&mut module);
    assert!(matches!(
        module.functions[&id].values[&checked].kind,
        HirValueKind::Constant(HirConstant::Bool(true))
    ));
    assert!(matches!(
        module.functions[&id].values[&checked_input].kind,
        HirValueKind::Instruction
    ));
}

#[test]
fn a_nullable_backedge_does_not_inherit_the_initial_allocation_fact() {
    let (f, header, current, checked, _) = fixture(true);
    let layout = osr::osr_layout(&f, header).unwrap();
    let index = layout
        .live_ins
        .iter()
        .position(|id| *id == current)
        .unwrap();
    let outlined = osr::outline(
        &f,
        &layout,
        HirId::new(),
        InternedString::new_global("nullable"),
    )
    .unwrap();
    assert!(!outlined.function.signature.params[index].attributes.nonnull);
    let mut module = HirModule::new(InternedString::new_global("facts"));
    let id = outlined.function.id;
    module.functions.insert(id, outlined.function);
    sign_fold::run_module(&mut module);
    assert!(matches!(
        module.functions[&id].values[&checked].kind,
        HirValueKind::Instruction
    ));
}
