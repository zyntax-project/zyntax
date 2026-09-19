//! A tuple carried around a loop needs no storage.
//!
//! `p = (p.1, p.0 + p.1)` builds a new aggregate every iteration and
//! merges it with the last at the header. After scalarization the loop
//! carries two integers: no `insertvalue`, no struct-typed phi. Where the
//! whole tuple is handed to a callee, it is rebuilt just there.
//!
//! Both programs run on Cranelift and on the interpreter after the pass,
//! since the pass changes what the backends see.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::aggregate_scalarize;
use zyntax_compiler::hir::{
    BinaryOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirParam, HirPhi, HirStructType, HirTerminator, HirType, HirValue,
    HirValueKind, ParamAttributes,
};
use zyntax_typed_ast::InternedString;

fn pair_ty() -> HirType {
    HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Pair")),
        fields: vec![HirType::I64, HirType::I64],
        packed: false,
    })
}

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

fn value(func: &mut HirFunction, ty: HirType) -> HirId {
    add_value(func, ty, HirValueKind::Instruction)
}

fn insert(result: HirId, agg: HirId, val: HirId, i: u32) -> HirInstruction {
    HirInstruction::InsertValue {
        result,
        ty: pair_ty(),
        aggregate: agg,
        value: val,
        indices: vec![i],
    }
}

fn extract(result: HirId, agg: HirId, i: u32) -> HirInstruction {
    HirInstruction::ExtractValue {
        result,
        ty: HirType::I64,
        aggregate: agg,
        indices: vec![i],
    }
}

fn arith(op: BinaryOp, result: HirId, left: HirId, right: HirId) -> HirInstruction {
    HirInstruction::Binary {
        op,
        result,
        ty: HirType::I64,
        left,
        right,
    }
}

/// `p = (1, 10); for 5 times: p = (p.1, p.0 + p.1); return p.0*1000 + p.1`.
/// With `callee`, each iteration also adds `callee(p)` into an
/// accumulator that is returned instead.
fn build_main(callee: Option<HirId>) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("main"),
        sig(vec![], vec![HirType::I64]),
    );
    let one = konst(&mut f, 1);
    let ten = konst(&mut f, 10);
    let zero = konst(&mut f, 0);
    let five = konst(&mut f, 5);
    let k1000 = konst(&mut f, 1000);

    let p0_undef = add_value(&mut f, pair_ty(), HirValueKind::Undef);
    let p0_a = value(&mut f, pair_ty());
    let p0 = value(&mut f, pair_ty());
    let p = value(&mut f, pair_ty());
    let i = value(&mut f, HirType::I64);
    let acc = value(&mut f, HirType::I64);
    let cond = value(&mut f, HirType::Bool);
    let a = value(&mut f, HirType::I64);
    let b = value(&mut f, HirType::I64);
    let s = value(&mut f, HirType::I64);
    let n_undef = add_value(&mut f, pair_ty(), HirValueKind::Undef);
    let n_a = value(&mut f, pair_ty());
    let next = value(&mut f, pair_ty());
    let i2 = value(&mut f, HirType::I64);
    let call_result = value(&mut f, HirType::I64);
    let acc2 = value(&mut f, HirType::I64);
    let ra = value(&mut f, HirType::I64);
    let rb = value(&mut f, HirType::I64);
    let t = value(&mut f, HirType::I64);
    let r = value(&mut f, HirType::I64);

    let entry = f.entry_block;
    let header = f.create_block();
    let body = f.create_block();
    let exit = f.create_block();

    {
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.instructions.push(insert(p0_a, p0_undef, one, 0));
        blk.instructions.push(insert(p0, p0_a, ten, 1));
        blk.terminator = HirTerminator::Branch { target: header };
    }
    {
        let blk = f.blocks.get_mut(&header).unwrap();
        blk.phis.push(HirPhi {
            result: p,
            ty: pair_ty(),
            incoming: vec![(p0, entry), (next, body)],
        });
        blk.phis.push(HirPhi {
            result: i,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (i2, body)],
        });
        blk.phis.push(HirPhi {
            result: acc,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (acc2, body)],
        });
        blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Lt,
            result: cond,
            ty: HirType::Bool,
            left: i,
            right: five,
        });
        blk.terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        };
    }
    {
        let blk = f.blocks.get_mut(&body).unwrap();
        if let Some(callee) = callee {
            blk.instructions.push(HirInstruction::Call {
                result: Some(call_result),
                callee: HirCallable::Function(callee),
                args: vec![p],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
            blk.instructions
                .push(arith(BinaryOp::Add, acc2, acc, call_result));
        } else {
            blk.instructions.push(arith(BinaryOp::Add, acc2, acc, zero));
        }
        blk.instructions.push(extract(a, p, 0));
        blk.instructions.push(extract(b, p, 1));
        blk.instructions.push(arith(BinaryOp::Add, s, a, b));
        blk.instructions.push(insert(n_a, n_undef, b, 0));
        blk.instructions.push(insert(next, n_a, s, 1));
        blk.instructions.push(arith(BinaryOp::Add, i2, i, one));
        blk.terminator = HirTerminator::Branch { target: header };
    }
    {
        let blk = f.blocks.get_mut(&exit).unwrap();
        blk.instructions.push(extract(ra, p, 0));
        blk.instructions.push(extract(rb, p, 1));
        blk.instructions.push(arith(BinaryOp::Mul, t, ra, k1000));
        blk.instructions.push(arith(BinaryOp::Add, r, t, rb));
        let ret = if callee.is_some() { acc } else { r };
        blk.terminator = HirTerminator::Return { values: vec![ret] };
    }
    f
}

/// `def sum(p: Pair): i64 { return p.0 + p.1 }`
fn build_sum() -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global("sum"),
        sig(vec![pair_ty()], vec![HirType::I64]),
    );
    let p = add_value(&mut f, pair_ty(), HirValueKind::Parameter(0));
    let a = value(&mut f, HirType::I64);
    let b = value(&mut f, HirType::I64);
    let s = value(&mut f, HirType::I64);
    let entry = f.entry_block;
    let blk: &mut HirBlock = f.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(extract(a, p, 0));
    blk.instructions.push(extract(b, p, 1));
    blk.instructions.push(arith(BinaryOp::Add, s, a, b));
    blk.terminator = HirTerminator::Return { values: vec![s] };
    f
}

/// (1,10) -> (10,11) -> (11,21) -> (21,32) -> (32,53) -> (53,85):
/// 53*1000 + 85.
const LOOP_EXPECTED: i64 = 53085;
/// The sums of the five tuples handed to `sum`: 11+21+32+53+85.
const CALL_EXPECTED: i64 = 202;

fn loop_module() -> (HirModule, HirId) {
    let main = build_main(None);
    let id = main.id;
    let mut module = HirModule::new(InternedString::new_global("scalarize_loop"));
    module.functions.insert(id, main);
    (module, id)
}

fn call_module() -> (HirModule, HirId) {
    let sum = build_sum();
    let sum_id = sum.id;
    let main = build_main(Some(sum_id));
    let id = main.id;
    let mut module = HirModule::new(InternedString::new_global("scalarize_call"));
    module.functions.insert(sum_id, sum);
    module.functions.insert(id, main);
    (module, id)
}

fn struct_phis(module: &HirModule, id: HirId) -> usize {
    module.functions[&id]
        .blocks
        .values()
        .flat_map(|b| b.phis.iter())
        .filter(|p| matches!(p.ty, HirType::Struct(_)))
        .count()
}

fn inserts(module: &HirModule, id: HirId) -> usize {
    module.functions[&id]
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i, HirInstruction::InsertValue { .. }))
        .count()
}

fn run_cranelift(module: &HirModule, id: HirId) -> i64 {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(id).expect("main compiled");
    let f: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
    unsafe { f() }
}

fn run_interp(module: &HirModule) -> i64 {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    let mut interp = HirInterpreter::new();
    let result = interp.call(module, "main", vec![]).expect("run");
    value_to_i64(&result).expect("an integer")
}

#[test]
fn a_loop_carried_tuple_becomes_two_scalars() {
    let (mut module, id) = loop_module();
    assert_eq!(run_cranelift(&module, id), LOOP_EXPECTED);
    assert_eq!(struct_phis(&module, id), 1);

    let stats = aggregate_scalarize::run_module(&mut module);
    assert!(stats.webs >= 1, "{stats:?}");
    assert_eq!(struct_phis(&module, id), 0, "the aggregate phi is gone");
    assert_eq!(inserts(&module, id), 0, "nothing builds the tuple any more");
    assert_eq!(stats.rematerialized, 0);

    assert_eq!(run_cranelift(&module, id), LOOP_EXPECTED);
    assert_eq!(run_interp(&module), LOOP_EXPECTED);
}

#[test]
fn a_tuple_handed_to_a_callee_is_rebuilt_there() {
    let (mut module, id) = call_module();
    assert_eq!(run_cranelift(&module, id), CALL_EXPECTED);

    let stats = aggregate_scalarize::run_module(&mut module);
    assert!(stats.webs >= 1, "{stats:?}");
    assert_eq!(struct_phis(&module, id), 0, "the aggregate phi is gone");
    assert_eq!(stats.rematerialized, 1, "once, at the call");
    assert_eq!(inserts(&module, id), 2, "the two fields of the argument");

    assert_eq!(run_cranelift(&module, id), CALL_EXPECTED);
    assert_eq!(run_interp(&module), CALL_EXPECTED);
}
