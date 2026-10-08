#![cfg(feature = "cranelift-backend")]
mod common;
use zyntax_compiler::{bounded_div, hir::*, hir_interp::HirInterpreter, value::ZyntaxValue};
use zyntax_typed_ast::InternedString;

fn constant(f: &mut HirFunction, n: i64) -> HirId {
    f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(n)))
}
fn binary(f: &mut HirFunction, block: HirId, op: BinaryOp, left: HirId, right: HirId) -> HirId {
    let result = f.create_value(HirType::I64, HirValueKind::Instruction);
    f.blocks
        .get_mut(&block)
        .unwrap()
        .instructions
        .push(HirInstruction::Binary {
            result,
            op,
            ty: HirType::I64,
            left,
            right,
        });
    result
}
fn function() -> HirFunction {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![HirType::I64];
    let mut f = HirFunction::new(InternedString::new_global("bounded"), sig);
    let id = f.create_value(HirType::I64, HirValueKind::Parameter(0));
    f.signature.params.push(HirParam {
        id,
        name: InternedString::new_global("x"),
        ty: HirType::I64,
        attributes: Default::default(),
        ownership: Default::default(),
    });
    f
}
fn check(f: HirFunction, samples: &[(i64, i64)]) {
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("bounds"));
    module.functions.insert(id, f);
    let mut interp = HirInterpreter::new();
    let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    clif.compile_module(&module).unwrap();
    clif.finalize_definitions().unwrap();
    // SAFETY: the fixture has a C (i64) -> i64 signature.
    let call: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
    #[cfg(feature = "llvm-backend")]
    let context = inkwell::context::Context::create();
    #[cfg(feature = "llvm-backend")]
    let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
    #[cfg(feature = "llvm-backend")]
    llvm.compile_module(&module).unwrap();
    for &(x, expected) in samples {
        assert!(
            matches!(interp.call(&module, "bounded", vec![ZyntaxValue::Int(x)]).unwrap(), ZyntaxValue::Int(n) if n == expected),
            "interpreter x={x} expected={expected}"
        );
        assert_eq!(call(x), expected, "Cranelift x={x}");
        #[cfg(feature = "llvm-backend")]
        {
            let call: extern "C" fn(i64) -> i64 =
                unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
            assert_eq!(call(x), expected, "LLVM x={x}");
        }
    }
}

#[test]
fn signed_remainders_and_division_match_in_every_tier() {
    for op in [BinaryOp::Div, BinaryOp::Rem] {
        for bound in [1_000_003, i32::MAX as i64 + 1] {
            let mut f = function();
            let entry = f.entry_block;
            let x = f.signature.params[0].id;
            let b = constant(&mut f, bound);
            let d = constant(&mut f, 7);
            let bounded = binary(&mut f, entry, BinaryOp::Rem, x, b);
            let value = binary(&mut f, entry, op, bounded, d);
            f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
                values: vec![value],
            };
            assert_eq!(bounded_div::run_function(&mut f), 1);
            assert_eq!(bounded_div::run_function(&mut f), 0, "idempotent");
            let samples: Vec<_> = [
                i64::MIN,
                i64::MIN + 1,
                i32::MIN as i64,
                -1_000_004,
                -7,
                -1,
                0,
                1,
                7,
                1_000_004,
                i32::MAX as i64,
                i64::MAX,
            ]
            .map(|x| {
                (
                    x,
                    if op == BinaryOp::Div {
                        (x % bound) / 7
                    } else {
                        (x % bound) % 7
                    },
                )
            })
            .into();
            check(f, &samples);
        }
    }
}

#[test]
fn wide_or_wrapping_operands_and_exceptional_divisors_stay_wide() {
    for (factor, divisor) in [
        (3, 7),
        (i64::MAX, 7),
        (1, -1),
        (1, 0),
        (1, i32::MAX as i64 + 1),
    ] {
        let mut f = function();
        let entry = f.entry_block;
        let x = f.signature.params[0].id;
        let bound = constant(&mut f, i32::MAX as i64);
        let k = constant(&mut f, factor);
        let d = constant(&mut f, divisor);
        let bounded = binary(&mut f, entry, BinaryOp::Rem, x, bound);
        let scaled = binary(&mut f, entry, BinaryOp::Mul, bounded, k);
        let value = binary(&mut f, entry, BinaryOp::Div, scaled, d);
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![value],
        };
        assert_eq!(bounded_div::run_function(&mut f), 0);
        if divisor > 0 {
            let samples: Vec<_> = [i64::MIN, -1_000_004, -1, 0, 1, 1_000_004, i64::MAX]
                .map(|x| (x, (x % (i32::MAX as i64)).wrapping_mul(factor) / divisor))
                .into();
            check(f, &samples);
        }
    }
}

#[test]
fn loop_carried_ranges_include_every_incoming_value() {
    for unknown_seed in [false, true] {
        let mut f = function();
        let entry = f.entry_block;
        let header = f.create_block();
        let body = f.create_block();
        let exit = f.create_block();
        let x = f.signature.params[0].id;
        let bound = constant(&mut f, 1_000_003);
        let zero = constant(&mut f, 0);
        let one = constant(&mut f, 1);
        let three = constant(&mut f, 3);
        let seven = constant(&mut f, 7);
        let count = constant(&mut f, 40);
        let acc = f.create_value(HirType::I64, HirValueKind::Instruction);
        let index = f.create_value(HirType::I64, HirValueKind::Instruction);
        let seed = if unknown_seed {
            x
        } else {
            binary(&mut f, entry, BinaryOp::Rem, x, bound)
        };
        let choice = binary(&mut f, body, BinaryOp::Rem, acc, seven);
        let scaled = binary(&mut f, body, BinaryOp::Mul, acc, three);
        let added = binary(&mut f, body, BinaryOp::Add, scaled, choice);
        let next = binary(&mut f, body, BinaryOp::Rem, added, bound);
        let next_index = binary(&mut f, body, BinaryOp::Add, index, one);
        let condition = f.create_value(HirType::Bool, HirValueKind::Instruction);
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
        let b = f.blocks.get_mut(&header).unwrap();
        b.phis = vec![
            HirPhi {
                result: acc,
                ty: HirType::I64,
                incoming: vec![(seed, entry), (next, body)],
            },
            HirPhi {
                result: index,
                ty: HirType::I64,
                incoming: vec![(zero, entry), (next_index, body)],
            },
        ];
        b.instructions.push(HirInstruction::Binary {
            result: condition,
            op: BinaryOp::Lt,
            ty: HirType::I64,
            left: index,
            right: count,
        });
        b.terminator = HirTerminator::CondBranch {
            condition,
            true_target: body,
            false_target: exit,
        };
        f.blocks.get_mut(&body).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![acc] };
        for (id, pred, succ) in [
            (entry, vec![], vec![header]),
            (header, vec![entry, body], vec![body, exit]),
            (body, vec![header], vec![header]),
            (exit, vec![header], vec![]),
        ] {
            let b = f.blocks.get_mut(&id).unwrap();
            b.predecessors = pred;
            b.successors = succ;
        }
        assert_eq!(
            bounded_div::run_function(&mut f),
            if unknown_seed { 0 } else { 2 }
        );
        let samples: Vec<_> = [i64::MIN, -1_000_004, -1, 0, 1, 1_000_004, i64::MAX]
            .map(|x| {
                let mut acc = if unknown_seed { x } else { x % 1_000_003 };
                for _ in 0..40 {
                    acc = acc.wrapping_mul(3).wrapping_add(acc % 7) % 1_000_003;
                }
                (x, acc)
            })
            .into();
        check(f, &samples);
    }
}

#[test]
fn sign_extended_i32_extremes_keep_signed_results() {
    for op in [BinaryOp::Div, BinaryOp::Rem] {
        for d in [2, 7, i32::MAX as i64] {
            let mut f = function();
            let entry = f.entry_block;
            let x = f.signature.params[0].id;
            let small = f.create_value(HirType::I32, HirValueKind::Instruction);
            let wide = f.create_value(HirType::I64, HirValueKind::Instruction);
            let divisor = constant(&mut f, d);
            f.blocks.get_mut(&entry).unwrap().instructions.extend([
                HirInstruction::Cast {
                    result: small,
                    op: CastOp::Trunc,
                    ty: HirType::I32,
                    operand: x,
                },
                HirInstruction::Cast {
                    result: wide,
                    op: CastOp::SExt,
                    ty: HirType::I64,
                    operand: small,
                },
            ]);
            let result = binary(&mut f, entry, op, wide, divisor);
            f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
                values: vec![result],
            };
            assert_eq!(bounded_div::run_function(&mut f), usize::from(d != 2));
            let samples: Vec<_> = [
                i64::MIN,
                i32::MIN as i64,
                -1,
                0,
                1,
                i32::MAX as i64,
                i64::MAX,
            ]
            .map(|x| {
                let n = x as i32 as i64;
                (x, if op == BinaryOp::Div { n / d } else { n % d })
            })
            .into();
            check(f, &samples);
        }
    }
}
