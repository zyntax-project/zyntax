//! Impossible integer tags disappear without changing calls or float payloads.
#![cfg(feature = "cranelift-backend")]
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use zyntax_compiler::{const_fold, hir::*};
use zyntax_typed_ast::InternedString;

static CALLS: AtomicUsize = AtomicUsize::new(0);
extern "C" fn observe(tag: i64) -> i64 {
    CALLS.fetch_add(1, Ordering::Relaxed);
    tag
}

fn build(bitop: BinaryOp, mask: i64, rhs: i64, op: BinaryOp, swap: bool) -> (HirModule, HirId) {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![HirType::F64];
    let mut f = HirFunction::new(InternedString::new_global("test_bits"), sig.clone());
    for (i, ty) in [HirType::I64, HirType::F64].into_iter().enumerate() {
        let id = f.create_value(ty.clone(), HirValueKind::Parameter(i as u32));
        f.signature.params.push(HirParam {
            id,
            name: InternedString::new_global("arg"),
            ty,
            attributes: Default::default(),
            ownership: Default::default(),
        });
    }
    sig.params = vec![f.signature.params[0].clone()];
    sig.returns = vec![HirType::I64];
    sig.is_pure = false;
    let mut host = HirFunction::new(InternedString::new_global("known_bits_observe"), sig);
    host.is_external = true;
    let tag = f.signature.params[0].id;
    let payload = f.signature.params[1].id;
    let observed = f.create_value(HirType::I64, HirValueKind::Instruction);
    let k = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(mask)));
    let r = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(rhs)));
    let masked = f.create_value(HirType::I64, HirValueKind::Instruction);
    let comparison = f.create_value(HirType::Bool, HirValueKind::Instruction);
    let fallback = f.create_value(HirType::F64, HirValueKind::Constant(HirConstant::F64(37.0)));
    let selected = f.create_value(HirType::F64, HirValueKind::Instruction);
    let entry = f.blocks.get_mut(&f.entry_block).unwrap();
    entry.instructions = vec![
        HirInstruction::Call {
            result: Some(observed),
            callee: HirCallable::Function(host.id),
            args: vec![tag],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
        HirInstruction::Binary {
            result: masked,
            op: bitop,
            ty: HirType::I64,
            left: observed,
            right: k,
        },
        HirInstruction::Binary {
            result: comparison,
            op,
            ty: HirType::I64,
            left: if swap { r } else { masked },
            right: if swap { masked } else { r },
        },
        HirInstruction::Select {
            result: selected,
            ty: HirType::F64,
            condition: comparison,
            true_val: payload,
            false_val: fallback,
        },
    ];
    entry.terminator = HirTerminator::Return {
        values: vec![selected],
    };
    let id = f.id;
    let mut module = HirModule::new(InternedString::new_global("known_bits"));
    module.functions.insert(host.id, host);
    module.functions.insert(id, f);
    (module, id)
}

fn check(mut run: impl FnMut(i64, f64) -> f64, bitop: BinaryOp, mask: i64, rhs: i64, op: BinaryOp) {
    for tag in [i64::MIN, -1, 0, 3, 4, 7, i64::MAX] {
        let value = match bitop {
            BinaryOp::Or => tag | mask,
            BinaryOp::And => tag & mask,
            BinaryOp::Xor => tag ^ mask,
            _ => unreachable!(),
        };
        let yes = if op == BinaryOp::Eq {
            value == rhs
        } else {
            value != rhs
        };
        for bits in [
            0,
            1 << 63,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x7ff8000000000001,
            0xfff8000000000042,
            1.25f64.to_bits(),
        ] {
            let input = f64::from_bits(bits);
            let before = CALLS.load(Ordering::Relaxed);
            assert_eq!(
                run(tag, input).to_bits(),
                if yes { bits } else { 37.0f64.to_bits() }
            );
            assert_eq!(
                CALLS.load(Ordering::Relaxed),
                before + 1,
                "producer must run once"
            );
        }
    }
}

#[test]
fn interpreter_and_jits_preserve_effects_and_float_selection() {
    use zyntax_compiler::hir_interp::{HirInterpreter, value_to_f64};
    use zyntax_compiler::value::ZyntaxValue;
    for (bitop, mask, rhs, decided) in [
        (BinaryOp::Or, 4, 3, true),
        (BinaryOp::And, 3, 4, true),
        (BinaryOp::Or, i64::MIN, 3, true),
        (BinaryOp::And, 0, 0, true),
        (BinaryOp::Or, 4, 7, false),
        (BinaryOp::Xor, 4, 3, false),
    ] {
        for op in [BinaryOp::Eq, BinaryOp::Ne] {
            for swap in [false, true] {
                for optimize in [false, true] {
                    let (mut module, id) = build(bitop, mask, rhs, op, swap);
                    if optimize {
                        assert_eq!(const_fold::fold_module(&mut module).folded > 0, decided);
                    }
                    let mut interp = HirInterpreter::new();
                    interp.register_symbol("known_bits_observe", observe as *const u8, 1);
                    check(
                        |tag, input| {
                            value_to_f64(
                                &interp
                                    .call(
                                        &module,
                                        "test_bits",
                                        vec![ZyntaxValue::Int(tag), ZyntaxValue::Float(input)],
                                    )
                                    .unwrap(),
                            )
                            .unwrap()
                        },
                        bitop,
                        mask,
                        rhs,
                        op,
                    );
                    let mut clif =
                        zyntax_compiler::cranelift_backend::CraneliftBackend::with_runtime_symbols(
                            &[("known_bits_observe", observe as *const u8)],
                        )
                        .unwrap();
                    clif.compile_module(&module).unwrap();
                    clif.finalize_definitions().unwrap();
                    // SAFETY: the fixture uses the C (i64, f64) -> f64 signature.
                    let call: unsafe extern "C" fn(i64, f64) -> f64 =
                        unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
                    check(
                        |tag, input| unsafe { call(tag, input) },
                        bitop,
                        mask,
                        rhs,
                        op,
                    );
                    #[cfg(feature = "llvm-backend")]
                    {
                        let context = inkwell::context::Context::create();
                        let mut llvm =
                            zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context)
                                .unwrap();
                        llvm.register_symbol("known_bits_observe", observe as *const u8);
                        llvm.compile_module(&module).unwrap();
                        // SAFETY: same fixture signature, with the backend kept alive.
                        let call: unsafe extern "C" fn(i64, f64) -> f64 =
                            unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
                        check(
                            |tag, input| unsafe { call(tag, input) },
                            bitop,
                            mask,
                            rhs,
                            op,
                        );
                    }
                }
            }
        }
    }
}
