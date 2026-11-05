#![cfg(feature = "cranelift-backend")]

use std::sync::Arc;
use zyntax_compiler::{hir::*, scalar_replace_alloc};
use zyntax_typed_ast::InternedString;

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

fn fixture() -> (HirModule, HirId, HirId) {
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
    let exit = f.create_block();
    let ptr = HirType::Ptr(Arc::new(HirType::I64));
    let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
    let seven = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(7)));
    let pointee = f.create_value(ptr.clone(), HirValueKind::Instruction);
    let slot = f.create_value(
        HirType::Ptr(Arc::new(ptr.clone())),
        HirValueKind::Instruction,
    );
    let alias = f.create_value(ptr.clone(), HirValueKind::Instruction);
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    f.blocks[&entry].instructions.extend([
        call(Some(pointee), Intrinsic::Malloc, vec![size]),
        HirInstruction::Store {
            value: seven,
            ptr: pointee,
            align: 8,
            volatile: false,
        },
        call(Some(slot), Intrinsic::Malloc, vec![size]),
        HirInstruction::Store {
            value: pointee,
            ptr: slot,
            align: 8,
            volatile: false,
        },
        HirInstruction::Load {
            result: alias,
            ptr: slot,
            ty: ptr,
            align: 8,
            volatile: false,
        },
        call(None, Intrinsic::Free, vec![slot]),
    ]);
    f.blocks[&entry].terminator = HirTerminator::Branch { target: exit };
    f.blocks[&exit].instructions.extend([
        HirInstruction::Load {
            result,
            ptr: alias,
            ty: HirType::I64,
            align: 8,
            volatile: false,
        },
        call(None, Intrinsic::Free, vec![alias]),
    ]);
    f.blocks[&exit].terminator = HirTerminator::Return {
        values: vec![result],
    };
    f.rebuild_cfg_edges();
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("forwarded_pointer"));
    module.functions.insert(id, f);
    (module, id, pointee)
}

fn execute(module: &HirModule, id: HirId) -> i64 {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let mut backend = CraneliftBackend::new().unwrap();
    backend.compile_module(module).unwrap();
    backend.finalize_definitions().unwrap();
    // SAFETY: the fixture's entry has no parameters and returns i64.
    let call: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(backend.get_function_ptr(id).unwrap()) };
    call()
}

#[test]
fn forwarding_a_pointer_rechecks_its_new_cross_block_uses() {
    let (mut module, id, pointee) = fixture();
    assert_eq!(execute(&module, id), 7);
    let stats = scalar_replace_alloc::run_module(&mut module);
    assert_eq!(stats.mallocs_eliminated, 1);
    assert_eq!(stats.frees_eliminated, 1);
    let remaining: Vec<_> = module.functions[&id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter_map(|inst| match inst {
            HirInstruction::Call {
                result,
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                ..
            } => *result,
            _ => None,
        })
        .collect();
    assert_eq!(remaining, vec![pointee]);
    assert_eq!(execute(&module, id), 7);
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    assert_eq!(
        value_to_i64(&HirInterpreter::new().call(&module, "main", vec![]).unwrap()).unwrap(),
        7
    );
    let original = bincode::serialize(&module).unwrap();
    assert_eq!(
        scalar_replace_alloc::run_module(&mut module).mallocs_eliminated,
        0
    );
    assert_eq!(bincode::serialize(&module).unwrap(), original);
}
