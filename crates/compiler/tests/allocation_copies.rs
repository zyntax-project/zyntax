//! Compiler preparation avoids unnecessary body copies and analysis indexes.
#![cfg(feature = "allocation-audit")]

use zyntax_compiler::allocation_audit::{CountingAllocator, Phase, Scope, for_phase};
use zyntax_compiler::bytecode::LazyModule;
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind,
};
use zyntax_typed_ast::InternedString;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static AUDIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

#[test]
fn scalarization_without_allocations_builds_no_indexes() {
    let _lock = AUDIT_LOCK.lock().unwrap();
    use zyntax_compiler::hir::{CastOp, HirPhi};
    let mut f = constant_function("scalar_arithmetic", 1000);
    let entry = f.entry_block;
    let exit = f.create_block();
    let ptr_ty = HirType::Ptr(std::sync::Arc::new(HirType::I64));
    let null = f.create_value(
        ptr_ty.clone(),
        HirValueKind::Constant(HirConstant::Null(ptr_ty.clone())),
    );
    let cast = f.create_value(ptr_ty.clone(), HirValueKind::Instruction);
    f.blocks[&entry].instructions.push(HirInstruction::Cast {
        result: cast,
        ty: ptr_ty.clone(),
        op: CastOp::Bitcast,
        operand: null,
    });
    let merged = f.create_value(ptr_ty.clone(), HirValueKind::Instruction);
    f.blocks[&exit].phis.push(HirPhi {
        result: merged,
        ty: ptr_ty,
        incoming: vec![(cast, entry)],
    });
    f.blocks[&exit].terminator = f.blocks[&entry].terminator.clone();
    f.blocks[&entry].terminator = HirTerminator::Branch { target: exit };
    f.rebuild_cfg_edges();
    let mut module = HirModule::new(InternedString::new_global("no_allocations"));
    module.functions.insert(f.id, f);
    let original = bincode::serialize(&module).unwrap();
    let before = for_phase(Phase::Optimize);
    {
        let _scope = Scope::enter(Phase::Optimize);
        for _ in 0..4 {
            assert_eq!(zyntax_compiler::heap_scalarize::run_module(&mut module), 0);
            assert_eq!(
                zyntax_compiler::scalar_replace_alloc::run_module(&mut module),
                Default::default()
            );
        }
    }
    let after = for_phase(Phase::Optimize);
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.reallocations, before.reallocations);
    assert_eq!(bincode::serialize(&module).unwrap(), original);
}

#[test]
fn locally_escaping_allocations_do_not_index_the_rest_of_the_function() {
    use zyntax_compiler::hir::{HirCallable, Intrinsic};
    let _lock = AUDIT_LOCK.lock().unwrap();
    let measure = |instructions| {
        let mut f = constant_function("escaping_allocation", instructions);
        let entry = f.entry_block;
        let exit = f.create_block();
        let arithmetic = std::mem::take(&mut f.blocks[&entry].instructions);
        f.blocks[&exit].instructions = arithmetic;
        f.blocks[&exit].terminator = f.blocks[&entry].terminator.clone();
        f.blocks[&entry].terminator = HirTerminator::Branch { target: exit };
        let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
        let ptr = f.create_value(
            HirType::Ptr(std::sync::Arc::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        for (result, callee, args) in [
            (
                Some(ptr),
                HirCallable::Intrinsic(Intrinsic::Malloc),
                vec![size],
            ),
            (None, HirCallable::Symbol("observe".into()), vec![ptr]),
            (None, HirCallable::Intrinsic(Intrinsic::Free), vec![ptr]),
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
        f.rebuild_cfg_edges();
        let mut module = HirModule::new(InternedString::new_global("escape_audit"));
        module.functions.insert(f.id, f);
        let original = bincode::serialize(&module).unwrap();
        let before = for_phase(Phase::Optimize);
        let stats = {
            let _scope = Scope::enter(Phase::Optimize);
            zyntax_compiler::scalar_replace_alloc::run_module(&mut module)
        };
        let after = for_phase(Phase::Optimize);
        assert_eq!(stats.candidates_examined, 1);
        assert_eq!(stats.escapes_skipped, 1);
        assert_eq!(stats.mallocs_eliminated, 0);
        assert_eq!(bincode::serialize(&module).unwrap(), original);
        (
            after.allocations - before.allocations,
            after.reallocations - before.reallocations,
            after.requested_bytes - before.requested_bytes,
        )
    };
    let small = measure(1);
    let large = measure(1000);
    eprintln!("local escape audit: small={small:?}, large={large:?}");
    assert_eq!(
        small, large,
        "unrelated uses must not grow analysis storage"
    );
}
