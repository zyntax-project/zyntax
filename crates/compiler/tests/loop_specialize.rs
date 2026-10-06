//! A tag can change before it enters a stable state. Both entries and
//! all live values must agree across the interpreter and native tiers.
#![cfg(feature = "cranelift-backend")]
use std::collections::HashSet;
use zyntax_compiler::hir::*;
use zyntax_compiler::loop_specialize;
use zyntax_typed_ast::InternedString;

fn sig(params: Vec<HirType>, returns: Vec<HirType>) -> HirFunctionSignature {
    HirFunctionSignature {
        params: params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| HirParam {
                id: HirId::new(),
                name: InternedString::new_global(&format!("p{}", i)),
                ty,
                attributes: ParamAttributes::default(),
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
    }
}

fn add_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn konst(func: &mut HirFunction, v: i64) -> HirId {
    add_value(
        func,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(v)),
    )
}

fn int(func: &mut HirFunction) -> HirId {
    add_value(func, HirType::I64, HirValueKind::Instruction)
}

fn bin(op: BinaryOp, result: HirId, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    }
}

fn build(toggles: bool) -> (HirModule, HirId) {
    let mut f = HirFunction::new(
        InternedString::new_global("f"),
        sig(vec![HirType::I64; 3], vec![HirType::I64]),
    );
    let params: Vec<_> = f
        .signature
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                p.id,
                HirValue {
                    id: p.id,
                    ty: p.ty.clone(),
                    kind: HirValueKind::Parameter(i as u32),
                    uses: HashSet::new(),
                    span: None,
                },
            )
        })
        .collect();
    for (id, v) in params {
        f.values.insert(id, v);
    }
    let tag_in = f.signature.params[0].id;
    let n = f.signature.params[1].id;
    let seed = f.signature.params[2].id;
    let entry = f.entry_block;
    let header = f.create_block();
    let body = f.create_block();
    let exit = f.create_block();
    let zero = konst(&mut f, 0);
    let one = konst(&mut f, 1);
    let two = konst(&mut f, 2);
    let ten = konst(&mut f, 10);
    let thousand = konst(&mut f, 1000);
    let million = konst(&mut f, 1000000);
    let tag = int(&mut f);
    let tag2 = int(&mut f);
    let i = int(&mut f);
    let i2 = int(&mut f);
    let acc = int(&mut f);
    let acc2 = int(&mut f);
    let more = add_value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let yes = add_value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let change = add_value(&mut f, HirType::Bool, HirValueKind::Instruction);
    let a = int(&mut f);
    let b = int(&mut f);
    let c = int(&mut f);
    let delta = int(&mut f);
    f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
    let h = f.blocks.get_mut(&header).unwrap();
    for (result, initial, next) in [(tag, tag_in, tag2), (i, zero, i2), (acc, seed, acc2)] {
        h.phis.push(HirPhi {
            result,
            ty: HirType::I64,
            incoming: vec![(initial, entry), (next, body)],
        });
    }
    h.instructions.push(bin(BinaryOp::Lt, more, i, n));
    h.terminator = HirTerminator::CondBranch {
        condition: more,
        true_target: body,
        false_target: exit,
    };
    let bdy = f.blocks.get_mut(&body).unwrap();
    bdy.instructions.extend([
        bin(BinaryOp::Eq, yes, tag, one),
        HirInstruction::Select {
            result: a,
            ty: HirType::I64,
            condition: yes,
            true_val: ten,
            false_val: one,
        },
        HirInstruction::Select {
            result: b,
            ty: HirType::I64,
            condition: yes,
            true_val: two,
            false_val: one,
        },
        bin(BinaryOp::Mul, c, a, b),
        bin(BinaryOp::Add, delta, c, one),
        bin(BinaryOp::Add, acc2, acc, delta),
        bin(BinaryOp::Eq, change, i, two),
        HirInstruction::Select {
            result: tag2,
            ty: HirType::I64,
            condition: if toggles { yes } else { change },
            true_val: if toggles { zero } else { one },
            false_val: if toggles { one } else { tag },
        },
        bin(BinaryOp::Add, i2, i, one),
    ]);
    bdy.terminator = HirTerminator::Branch { target: header };
    let t = int(&mut f);
    let u = int(&mut f);
    let v = int(&mut f);
    let result = int(&mut f);
    f.blocks.get_mut(&exit).unwrap().instructions.extend([
        bin(BinaryOp::Mul, t, tag, thousand),
        bin(BinaryOp::Mul, u, i, million),
        bin(BinaryOp::Add, v, acc, t),
        bin(BinaryOp::Add, result, v, u),
    ]);
    f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return {
        values: vec![result],
    };
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("stable_loop"));
    module.functions.insert(id, f);
    (module, id)
}

fn reference(mut tag: i64, n: i64, mut acc: i64) -> i64 {
    let mut i = 0;
    while i < n {
        acc += if tag == 1 { 21 } else { 2 };
        if i == 2 {
            tag = 1;
        }
        i += 1;
    }
    acc + tag * 1000 + i * 1000000
}

fn check(mut run: impl FnMut(i64, i64, i64) -> i64) {
    for tag in [0, 1, 2, -1] {
        for n in [-1, 0, 1, 2, 3, 4, 20] {
            for seed in [-7, 0, 19] {
                assert_eq!(
                    run(tag, n, seed),
                    reference(tag, n, seed),
                    "tag={tag}, n={n}, seed={seed}"
                );
            }
        }
    }
}

#[test]
fn guarded_entry_preserves_current_state_and_live_exits() {
    let (mut module, id) = build(false);
    assert_eq!(loop_specialize::run_module(&mut module), 1);
    assert_eq!(loop_specialize::run_module(&mut module), 0);
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;
    let mut interp = HirInterpreter::new();
    check(|tag, n, seed| {
        value_to_i64(
            &interp
                .call(
                    &module,
                    "f",
                    vec![
                        ZyntaxValue::Int(tag),
                        ZyntaxValue::Int(n),
                        ZyntaxValue::Int(seed),
                    ],
                )
                .unwrap(),
        )
        .unwrap()
    });
    let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    clif.compile_module(&module).unwrap();
    clif.finalize_definitions().unwrap();
    let f: unsafe extern "C" fn(i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
    check(|tag, n, seed| unsafe { f(tag, n, seed) });
    #[cfg(feature = "llvm-backend")]
    {
        let context = inkwell::context::Context::create();
        let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
        llvm.compile_module(&module).unwrap();
        let f: unsafe extern "C" fn(i64, i64, i64) -> i64 =
            unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
        check(|tag, n, seed| unsafe { f(tag, n, seed) });
    }
}

#[test]
fn a_tag_that_can_change_again_is_not_specialized() {
    let (mut module, _) = build(true);
    assert_eq!(loop_specialize::run_module(&mut module), 0);
}

#[test]
fn an_exit_phi_keeps_the_edge_that_bypasses_the_loop() {
    let (mut module, id) = build(false);
    let f = module.functions.get_mut(&id).unwrap();
    let entry = f.entry_block;
    let HirTerminator::Branch { target: header } = f.blocks[&entry].terminator else {
        panic!()
    };
    let HirTerminator::CondBranch {
        false_target: exit, ..
    } = f.blocks[&header].terminator
    else {
        panic!()
    };
    let n = f.signature.params[1].id;
    let seed = f.signature.params[2].id;
    let acc = f.blocks[&header].phis[2].result;
    let zero = konst(f, 0);
    let bypass = add_value(f, HirType::Bool, HirValueKind::Instruction);
    f.blocks
        .get_mut(&entry)
        .unwrap()
        .instructions
        .push(bin(BinaryOp::Lt, bypass, n, zero));
    f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::CondBranch {
        condition: bypass,
        true_target: exit,
        false_target: header,
    };
    let result = int(f);
    let block = f.blocks.get_mut(&exit).unwrap();
    block.instructions.clear();
    block.phis.push(HirPhi {
        result,
        ty: HirType::I64,
        incoming: vec![(seed, entry), (acc, header)],
    });
    block.terminator = HirTerminator::Return {
        values: vec![result],
    };
    assert_eq!(loop_specialize::run_module(&mut module), 1);
    let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    clif.compile_module(&module).unwrap();
    clif.finalize_definitions().unwrap();
    let f: unsafe extern "C" fn(i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
    for tag in [0, 1, 2] {
        for n in [-3, 0, 2, 3, 4, 12] {
            let final_tag = if n > 2 { 1 } else { tag };
            let expected = reference(tag, n, 17) - final_tag * 1000 - n.max(0) * 1000000;
            assert_eq!(unsafe { f(tag, n, 17) }, expected);
        }
    }
}
