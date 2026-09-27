//! Counted loops versioned on the bounds of their indices.
//!
//! `f(len, n)` walks `i` over `0..n`, forms `idx = i + OFFSET`, and
//! fails like a checked list access when `idx < 0` or `idx >= len`,
//! returning a code that says which check failed, at which iteration and
//! with what accumulated so far. Otherwise it adds `idx` to an
//! accumulator and returns it. Each program runs on the interpreter,
//! Cranelift and LLVM after the pipeline, with lengths above and below
//! what the loop needs, so both the copy and the checked loop run and
//! must agree with a plain reading of the program.

#![cfg(feature = "cranelift-backend")]

use std::collections::HashSet;
use zyntax_compiler::bounds_version;
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction, HirModule,
    HirParam, HirPhi, HirTerminator, HirType, HirValue, HirValueKind, ParamAttributes,
};
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

fn flag(func: &mut HirFunction) -> HirId {
    add_value(func, HirType::Bool, HirValueKind::Instruction)
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

/// How the loop reads its bound.
#[derive(Clone, Copy, PartialEq)]
enum Bound {
    /// `len` itself, read before the loop.
    Before,
    /// A length kept in memory, read every iteration, which the body
    /// shrinks by one each time round.
    Shrinking,
}

const NEG: i64 = -1;
const OOB: i64 = -2;

/// The code `f` returns for a failed check.
fn failed(code: i64, i: i64, acc: i64) -> i64 {
    code - (acc * 1000 + i)
}

/// What `f(len, n)` computes.
fn reference(offset: i64, bound: Bound, len: i64, n: i64) -> i64 {
    let mut acc = 0;
    let mut len = len;
    let mut i = 0;
    while i < n {
        let idx = i + offset;
        if idx < 0 {
            return failed(NEG, i, acc);
        }
        if idx >= len {
            return failed(OOB, i, acc);
        }
        acc += idx;
        if bound == Bound::Shrinking {
            len -= 1;
        }
        i += 1;
    }
    acc
}

fn build(offset: i64, bound: Bound) -> (HirModule, HirId) {
    let mut f = HirFunction::new(
        InternedString::new_global("f"),
        sig(vec![HirType::I64, HirType::I64], vec![HirType::I64]),
    );
    let len = add_value(&mut f, HirType::I64, HirValueKind::Parameter(0));
    let n = add_value(&mut f, HirType::I64, HirValueKind::Parameter(1));
    let zero = konst(&mut f, 0);
    let one = konst(&mut f, 1);
    let off = konst(&mut f, offset);
    let thousand = konst(&mut f, 1000);
    let neg_code = konst(&mut f, NEG);
    let oob_code = konst(&mut f, OOB);

    let entry = f.entry_block;
    let header = f.create_block();
    let body = f.create_block();
    let check = f.create_block();
    let ok = f.create_block();
    let fail_neg = f.create_block();
    let fail_oob = f.create_block();
    let exit = f.create_block();

    let slot = add_value(
        &mut f,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let i = int(&mut f);
    let acc = int(&mut f);
    let more = flag(&mut f);
    let idx = int(&mut f);
    let neg = flag(&mut f);
    let cur = int(&mut f);
    let oob = flag(&mut f);
    let acc2 = int(&mut f);
    let shrunk = int(&mut f);
    let i2 = int(&mut f);

    {
        let blk = f.blocks.get_mut(&entry).unwrap();
        if bound == Bound::Shrinking {
            blk.instructions.push(HirInstruction::Alloca {
                result: slot,
                ty: HirType::I64,
                count: None,
                align: 8,
            });
            blk.instructions.push(HirInstruction::Store {
                value: len,
                ptr: slot,
                align: 8,
                volatile: false,
            });
        }
        blk.terminator = HirTerminator::Branch { target: header };
    }
    {
        let blk = f.blocks.get_mut(&header).unwrap();
        blk.phis.push(HirPhi {
            result: i,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (i2, ok)],
        });
        blk.phis.push(HirPhi {
            result: acc,
            ty: HirType::I64,
            incoming: vec![(zero, entry), (acc2, ok)],
        });
        blk.instructions.push(bin(BinaryOp::Lt, more, i, n));
        blk.terminator = HirTerminator::CondBranch {
            condition: more,
            true_target: body,
            false_target: exit,
        };
    }
    {
        let blk = f.blocks.get_mut(&body).unwrap();
        blk.instructions.push(bin(BinaryOp::Add, idx, i, off));
        blk.instructions.push(bin(BinaryOp::Lt, neg, idx, zero));
        blk.terminator = HirTerminator::CondBranch {
            condition: neg,
            true_target: fail_neg,
            false_target: check,
        };
    }
    {
        let blk = f.blocks.get_mut(&check).unwrap();
        let limit = match bound {
            Bound::Before => len,
            Bound::Shrinking => {
                blk.instructions.push(HirInstruction::Load {
                    result: cur,
                    ty: HirType::I64,
                    ptr: slot,
                    align: 8,
                    volatile: false,
                });
                cur
            }
        };
        blk.instructions.push(bin(BinaryOp::Ge, oob, idx, limit));
        blk.terminator = HirTerminator::CondBranch {
            condition: oob,
            true_target: fail_oob,
            false_target: ok,
        };
    }
    {
        let blk = f.blocks.get_mut(&ok).unwrap();
        blk.instructions.push(bin(BinaryOp::Add, acc2, acc, idx));
        if bound == Bound::Shrinking {
            blk.instructions.push(bin(BinaryOp::Sub, shrunk, cur, one));
            blk.instructions.push(HirInstruction::Store {
                value: shrunk,
                ptr: slot,
                align: 8,
                volatile: false,
            });
        }
        blk.instructions.push(bin(BinaryOp::Add, i2, i, one));
        blk.terminator = HirTerminator::Branch { target: header };
    }
    for (blk_id, code) in [(fail_neg, neg_code), (fail_oob, oob_code)] {
        let t = int(&mut f);
        let t2 = int(&mut f);
        let r = int(&mut f);
        let blk = f.blocks.get_mut(&blk_id).unwrap();
        blk.instructions.push(bin(BinaryOp::Mul, t, acc, thousand));
        blk.instructions.push(bin(BinaryOp::Add, t2, t, i));
        blk.instructions.push(bin(BinaryOp::Sub, r, code, t2));
        blk.terminator = HirTerminator::Return { values: vec![r] };
    }
    f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![acc] };

    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("bounds_version"));
    module.functions.insert(id, f);
    (module, id)
}

fn optimised(offset: i64, bound: Bound) -> (HirModule, HirId) {
    let (mut module, id) = build(offset, bound);
    let stats = zyntax_compiler::run_interp_safe_opts(&mut module);
    // A bound the body changes leaves nothing the copy could decide
    // once the sign check is folded on its own.
    let expected = usize::from(bound == Bound::Before);
    assert_eq!(
        stats.bounds_version.versioned, expected,
        "the pipeline versions a loop over a bound read before it, only"
    );
    (module, id)
}

/// Lengths and trip counts that take the copy, the checked loop at
/// either check, and no iteration at all.
const CASES: &[(i64, i64)] = &[
    (10, 10),
    (10, 9),
    (10, 11),
    (10, 15),
    (10, 0),
    (0, 0),
    (0, 3),
    (5, -3),
    (3, i64::MAX),
];

fn check_all(offset: i64, bound: Bound, mut run: impl FnMut(i64, i64) -> i64, tier: &str) {
    for &(len, n) in CASES {
        assert_eq!(
            run(len, n),
            reference(offset, bound, len, n),
            "{tier}: offset {offset}, len {len}, n {n}"
        );
    }
}

fn run_interp(offset: i64, bound: Bound) {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_i64};
    use zyntax_compiler::value::ZyntaxValue;
    let (module, _) = optimised(offset, bound);
    let mut interp = HirInterpreter::new();
    let run = |len: i64, n: i64| {
        let v = interp
            .call(
                &module,
                "f",
                vec![ZyntaxValue::Int(len), ZyntaxValue::Int(n)],
            )
            .expect("run");
        value_to_i64(&v).expect("an integer")
    };
    check_all(offset, bound, run, "interpreter");
}

fn run_cranelift(offset: i64, bound: Bound) {
    use zyntax_compiler::cranelift_backend::CraneliftBackend;
    let (module, id) = optimised(offset, bound);
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.compile_module(&module).expect("compile");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(id).expect("f compiled");
    let f: unsafe extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    check_all(offset, bound, |len, n| unsafe { f(len, n) }, "cranelift");
}

#[cfg(feature = "llvm-backend")]
fn run_llvm(offset: i64, bound: Bound) {
    use inkwell::context::Context;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;
    if zyntax_compiler::llvm_link::find_linker().is_err() {
        eprintln!("no system linker; skipping the LLVM leg");
        return;
    }
    let (module, id) = optimised(offset, bound);
    let context = Context::create();
    let mut backend = LLVMJitBackend::new(&context).expect("backend");
    backend.compile_module(&module).expect("compile");
    let ptr = backend.get_function_pointer(id).expect("f compiled");
    let f: unsafe extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    check_all(offset, bound, |len, n| unsafe { f(len, n) }, "llvm");
}

fn run_all(offset: i64, bound: Bound) {
    run_interp(offset, bound);
    run_cranelift(offset, bound);
    #[cfg(feature = "llvm-backend")]
    run_llvm(offset, bound);
}

#[test]
fn a_loop_over_a_bound_read_before_it_gets_a_check_free_copy() {
    let (mut module, id) = build(0, Bound::Before);
    let f = module.functions.get_mut(&id).unwrap();
    let stats = bounds_version::run_function(f);
    assert_eq!(stats.versioned, 1);
    assert_eq!(stats.folded, 2, "both the sign and the length check");
    let again = bounds_version::run_function(f);
    assert_eq!(
        again.versioned, 0,
        "a versioned loop is not versioned again"
    );
}

#[test]
fn the_copy_and_the_checked_loop_agree_on_every_tier() {
    run_all(0, Bound::Before);
}

#[test]
fn an_index_past_the_counter_fails_at_the_same_iteration() {
    run_all(1, Bound::Before);
}

#[test]
fn a_negative_first_index_takes_the_checked_loop() {
    run_all(-1, Bound::Before);
}

#[test]
fn a_bound_the_body_changes_is_checked_every_iteration() {
    let (mut module, id) = build(0, Bound::Shrinking);
    let f = module.functions.get_mut(&id).unwrap();
    let stats = bounds_version::run_function(f);
    assert_eq!(stats.folded, 1, "only the sign check is decided");
    run_all(0, Bound::Shrinking);
}
