//! Availability must hold on every incoming edge, including loop backedges.
#![cfg(feature = "cranelift-backend")]
mod common;

use zyntax_compiler::{hir::*, hir_interp::HirInterpreter, load_cse, value::ZyntaxValue};
use zyntax_typed_ast::InternedString;

// Read p before a diamond and again at its join. The true arm may write q.
// The condition is a parameter so both paths execute through the same JIT code.
fn diamond(write: bool, distinct: bool) -> (HirModule, HirId, HirId, HirId) {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![HirType::I64];
    sig.is_pure = false;
    let mut f = HirFunction::new(InternedString::new_global("diamond"), sig);
    for (i, ty) in [
        HirType::Ptr(Box::new(HirType::I64)),
        HirType::Ptr(Box::new(HirType::I64)),
        HirType::I64,
    ]
    .into_iter()
    .enumerate()
    {
        let id = f.create_value(ty.clone(), HirValueKind::Parameter(i as u32));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global("arg"),
            ty,
            attributes: Default::default(),
            ownership: Default::default(),
        });
    }
    let p = f.signature.params[0].id;
    let q = f.signature.params[usize::from(distinct)].id;
    let condition_arg = f.signature.params[2].id;
    let condition = f.create_value(HirType::Bool, HirValueKind::Instruction);
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let first = f.create_value(HirType::I64, HirValueKind::Instruction);
    let last = f.create_value(HirType::I64, HirValueKind::Instruction);
    let value = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(73)));
    let yes = HirId::new();
    let no = HirId::new();
    let join = HirId::new();
    let load = |result| HirInstruction::Load {
        result,
        ty: HirType::I64,
        ptr: p,
        align: 8,
        volatile: false,
    };
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.instructions.push(HirInstruction::Binary {
        result: condition,
        op: BinaryOp::Ne,
        ty: HirType::Bool,
        left: condition_arg,
        right: zero,
    });
    entry.instructions.push(load(first));
    entry.terminator = HirTerminator::CondBranch {
        condition,
        true_target: yes,
        false_target: no,
    };
    let mut y = HirBlock::new(yes);
    if write {
        y.instructions.push(HirInstruction::Store {
            value,
            ptr: q,
            align: 8,
            volatile: false,
        });
    }
    y.terminator = HirTerminator::Branch { target: join };
    let mut n = HirBlock::new(no);
    n.terminator = HirTerminator::Branch { target: join };
    let mut j = HirBlock::new(join);
    j.instructions.push(load(last));
    j.terminator = HirTerminator::Return { values: vec![last] };
    // Deliberately non-topological insertion order and absent cached CFG edges.
    f.blocks.insert(join, j);
    f.blocks.insert(no, n);
    f.blocks.insert(yes, y);
    let id = f.id;
    let mut m = HirModule::new(InternedString::new_global("loads"));
    m.functions.insert(id, f);
    (m, id, yes, join)
}

fn success_path(write: bool, distinct: bool) -> (HirModule, HirId, HirId, HirId) {
    let (mut m, id, yes, join) = diamond(write, distinct);
    let f = m.functions.get_mut(&id).unwrap();
    let entry = &f.blocks[&f.entry_block];
    let no = match entry.terminator {
        HirTerminator::CondBranch { false_target, .. } => false_target,
        _ => unreachable!(),
    };
    let first = entry
        .instructions
        .iter()
        .find_map(|i| match i {
            HirInstruction::Load { result, .. } => Some(*result),
            _ => None,
        })
        .unwrap();
    f.blocks.get_mut(&no).unwrap().terminator = HirTerminator::Return {
        values: vec![first],
    };
    (m, id, yes, join)
}

fn check(mut call: impl FnMut(*mut i64, *mut i64, i64) -> i64, write: bool) {
    for initial in [-11, 0, 29] {
        for condition in [0, 1] {
            let mut memory = initial;
            let ptr = &mut memory as *mut i64;
            let expected = if write && condition != 0 { 73 } else { initial };
            assert_eq!(call(ptr, ptr, condition), expected);
            assert_eq!(memory, expected);
        }
    }
}

#[test]
fn one_arm_writes_and_aliased_parameters_execute_in_every_tier() {
    for write in [false, true] {
        for distinct in [false, true] {
            let (mut module, id, _, _) = success_path(write, distinct);
            assert_eq!(
                load_cse::run_regions(module.functions.get_mut(&id).unwrap()).eliminated,
                usize::from(!write)
            );
            let mut interp = HirInterpreter::new();
            check(
                |p, q, c| match interp
                    .call(
                        &module,
                        "diamond",
                        vec![
                            ZyntaxValue::Pointer(p.cast()),
                            ZyntaxValue::Pointer(q.cast()),
                            ZyntaxValue::Int(c),
                        ],
                    )
                    .unwrap()
                {
                    ZyntaxValue::Int(n) => n,
                    other => panic!("{other:?}"),
                },
                write,
            );
            let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
            clif.compile_module(&module).unwrap();
            clif.finalize_definitions().unwrap();
            // SAFETY: the fixture uses this C signature; its backend remains alive.
            let call: unsafe extern "C" fn(*mut i64, *mut i64, i64) -> i64 =
                unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
            check(|p, q, c| unsafe { call(p, q, c) }, write);
            #[cfg(feature = "llvm-backend")]
            {
                let context = inkwell::context::Context::create();
                let mut llvm =
                    zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
                llvm.compile_module(&module).unwrap();
                // SAFETY: same fixture signature, with the backend kept alive.
                let call: unsafe extern "C" fn(*mut i64, *mut i64, i64) -> i64 =
                    unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
                check(|p, q, c| unsafe { call(p, q, c) }, write);
            }
        }
    }
}

#[test]
fn loads_from_one_arm_or_different_arms_do_not_dominate_the_join() {
    for both in [false, true] {
        let (mut module, id, yes, _) = diamond(false, false);
        let f = module.functions.get_mut(&id).unwrap();
        let entry = f.blocks.get_mut(&f.entry_block).unwrap();
        let load = entry.instructions.pop().unwrap();
        let no = match entry.terminator {
            HirTerminator::CondBranch { false_target, .. } => false_target,
            _ => unreachable!(),
        };
        f.blocks
            .get_mut(&yes)
            .unwrap()
            .instructions
            .push(load.clone());
        if both {
            let mut load = load;
            if let HirInstruction::Load { result, .. } = &mut load {
                *result = f.create_value(HirType::I64, HirValueKind::Instruction);
            }
            f.blocks.get_mut(&no).unwrap().instructions.push(load);
        }
        assert_eq!(load_cse::run_regions(f).eliminated, 0);
    }
}

#[test]
fn calls_suspension_volatile_access_and_invoke_kill_incoming_loads() {
    for variant in 0..5 {
        let (mut module, id, yes, join) = success_path(false, false);
        let f = module.functions.get_mut(&id).unwrap();
        let ptr = f.signature.params[0].id;
        let result = f.create_value(HirType::I64, HirValueKind::Instruction);
        let block = f.blocks.get_mut(&yes).unwrap();
        match variant {
            0 => block.instructions.push(HirInstruction::Call {
                result: None,
                callee: HirCallable::Symbol("mutate".into()),
                args: vec![ptr],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            }),
            1 => block.instructions.push(HirInstruction::CallClosure {
                result: None,
                closure: ptr,
                args: vec![],
            }),
            2 => block
                .instructions
                .push(HirInstruction::FiberYield { value: ptr }),
            3 => block.instructions.push(HirInstruction::Load {
                result,
                ty: HirType::I64,
                ptr,
                align: 8,
                volatile: true,
            }),
            4 => {
                block.terminator = HirTerminator::Invoke {
                    callee: HirCallable::Symbol("mutate".into()),
                    args: vec![ptr],
                    normal: join,
                    unwind: join,
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(load_cse::run_regions(f).eliminated, 0, "variant={variant}");
    }
}

#[test]
fn loop_backedge_write_prevents_reusing_a_preloop_load() {
    let (mut module, id, yes, join) = diamond(true, true);
    let f = module.functions.get_mut(&id).unwrap();
    let exit = HirId::new();
    let old_entry_term = f.blocks[&f.entry_block].terminator.clone();
    let (condition, no) = match old_entry_term {
        HirTerminator::CondBranch {
            condition,
            false_target,
            ..
        } => (condition, false_target),
        _ => unreachable!(),
    };
    // entry -> header(load, if condition) -> write -> header, or exit.
    f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::Branch { target: join };
    f.blocks.get_mut(&no).unwrap().terminator = HirTerminator::Unreachable;
    let mut end = HirBlock::new(exit);
    end.terminator = f.blocks[&join].terminator.clone();
    f.blocks.get_mut(&join).unwrap().terminator = HirTerminator::CondBranch {
        condition,
        true_target: yes,
        false_target: exit,
    };
    f.blocks.insert(exit, end);
    assert_eq!(load_cse::run_regions(f).eliminated, 0);
}

#[test]
fn unchanged_length_makes_the_second_bounds_branch_redundant() {
    let (mut m, id, _, join) = success_path(false, false);
    let f = m.functions.get_mut(&id).unwrap();
    let first = match f.blocks[&f.entry_block].instructions[1] {
        HirInstruction::Load { result, .. } => result,
        _ => unreachable!(),
    };
    let last = match f.blocks[&join].instructions[0] {
        HirInstruction::Load { result, .. } => result,
        _ => unreachable!(),
    };
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    let mut compare = entry.instructions.remove(0);
    if let HirInstruction::Binary { left, .. } = &mut compare {
        *left = first;
    }
    entry.instructions.push(compare.clone());
    let second = f.create_value(HirType::Bool, HirValueKind::Instruction);
    if let HirInstruction::Binary { result, left, .. } = &mut compare {
        *result = second;
        *left = last;
    }
    let yes = f.create_block();
    let no = f.create_block();
    for target in [yes, no] {
        f.blocks.get_mut(&target).unwrap().terminator =
            HirTerminator::Return { values: vec![last] };
    }
    let block = f.blocks.get_mut(&join).unwrap();
    block.instructions.push(compare);
    block.terminator = HirTerminator::CondBranch {
        condition: second,
        true_target: yes,
        false_target: no,
    };
    assert_eq!(load_cse::run_regions(f).eliminated, 1);
    assert!(zyntax_compiler::cse::eliminate(f).eliminated > 0);
    assert_eq!(zyntax_compiler::branch_fold::run(f).folded, 1);
    assert!(
        matches!(f.blocks[&join].terminator, HirTerminator::Branch { target } if target == yes)
    );
}

#[test]
fn unchanged_join_still_starts_a_new_region() {
    let (mut m, id, _, _) = diamond(false, false);
    assert_eq!(
        load_cse::run_regions(m.functions.get_mut(&id).unwrap()).eliminated,
        0
    );
}
