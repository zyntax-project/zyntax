//! Read-only compiler paths do not copy entire function bodies.
#![cfg(feature = "allocation-audit")]

use zyntax_compiler::allocation_audit::{Phase, for_phase};
use zyntax_compiler::bytecode::LazyModule;
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind,
};
use zyntax_typed_ast::InternedString;

fn constant_function(name: &str, instructions: usize) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global(name),
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
    let one = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(1)));
    let mut last = one;
    for _ in 0..instructions {
        let result = f.create_value(HirType::I64, HirValueKind::Instruction);
        f.blocks[&f.entry_block]
            .instructions
            .push(HirInstruction::Binary {
                result,
                op: BinaryOp::Add,
                left: last,
                right: one,
                ty: HirType::I64,
            });
        last = result;
    }
    f.blocks[&f.entry_block].terminator = HirTerminator::Return { values: vec![last] };
    f
}

#[test]
fn rejected_inlining_and_borrowed_reads_do_not_clone_bodies() {
    let mut module = HirModule::new(InternedString::new_global("allocation_copies"));
    let small = constant_function("small", 1);
    let large = constant_function("large", 1000);
    let id = large.id;
    module.functions.insert(small.id, small);
    module.functions.insert(large.id, large);
    let before = for_phase(Phase::HirClone).entries;
    for _ in 0..2 {
        let stats = zyntax_compiler::inline::run_module_recursive(&mut module);
        assert_eq!(stats.skipped_too_large, 1);
        assert_eq!(stats.functions_visited, 0);
        assert_eq!(stats.self_calls_inlined, 0);
    }
    assert_eq!(for_phase(Phase::HirClone).entries, before);

    let source = LazyModule::eager(module);
    let original = &source.shell().functions[&id];
    assert_eq!(
        source.with_function(id, |f| {
            assert!(std::ptr::eq(f, original));
            f.blocks[&f.entry_block].instructions.len()
        }),
        Some(1000)
    );
    assert_eq!(for_phase(Phase::HirClone).entries, before);
}
